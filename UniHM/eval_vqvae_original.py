import argparse
import numpy as np
import torch

from UniHM.SFT.utils import build_seq_dataloaders, DECODER_KEY_ALIASES, ROBOT_KEYS_ORDER
from UniHM.vqvae import MultiDecoderVQVAE
from UniHM.vqvae.decoder import Decoder, MLPDecoder
from UniHM.metrics.common_metrics import mpjpe, fhlt, fhlr, fid, smoothness_l2, rollout_drift


def parse_int_list(s: str):
    return [int(x.strip()) for x in s.split(',') if x.strip()]


def parse_str_list(s: str):
    return [x.strip() for x in s.split(',') if x.strip()]


def build_vqvae_like_training(args, device: torch.device):
    """
    与 train_vqvae_mano.py / QwenVQVAE.py 保持一致：
    - 先构建 MultiDecoderVQVAE
    - 再挂接 mano_decoder
    - 使用 strict=False 加载完整 state_dict
    """
    model = MultiDecoderVQVAE(
        in_dim=args.in_dim,
        h_dim=args.h_dim,
        res_h_dim=args.res_h_dim,
        n_res_layers=args.n_res_layers,
        n_embeddings=args.n_embeddings,
        embedding_dim=args.embedding_dim,
        beta=args.beta,
        num_decoders=len(args.decoder_out_dims),
        decoder_out_channels=args.decoder_out_dims,
        use_mlp=args.use_mlp,
        input_length=args.input_length,
    )

    if args.use_mlp:
        mano_decoder = MLPDecoder(
            args.embedding_dim,
            args.h_dim,
            args.n_res_layers,
            args.res_h_dim,
            out_channels=args.mano_dim,
        )
    else:
        mano_decoder = Decoder(
            args.in_dim,
            args.h_dim,
            args.n_res_layers,
            args.res_h_dim,
            outdim=args.mano_dim,
            embedding_dim=args.embedding_dim,
        )
    model.mano_decoder = mano_decoder

    state = torch.load(args.ckpt, map_location='cpu')
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"Loaded ckpt: {args.ckpt}")
    print(f"Missing keys: {len(missing)} | Unexpected keys: {len(unexpected)}")

    model = model.to(device)
    model.eval()
    return model


def resolve_targets(targets, decoder_keys, device):
    ys = []
    for k in decoder_keys:
        aliases = DECODER_KEY_ALIASES.get(k, [k])
        kk = next((a for a in aliases if a in targets), None)
        ys.append(targets[kk].to(device) if kk is not None else None)
    return ys


def evaluate(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    args.decoder_out_dims = parse_int_list(args.decoder_out_dims)
    args.decoder_keys = parse_str_list(args.decoder_keys)

    model = build_vqvae_like_training(args, device)

    _, val_loader = build_seq_dataloaders(
        args.seq_glob,
        batch_size=args.batch_size,
        num_workers=max(1, args.num_workers),
    )

    metric_names = ["mpjpe_legacy", "qpos_mae", "trans_mae", "rot_mae", "fid", "smooth", "drift"]
    per_robot = {k: {m: [] for m in metric_names} for k in args.decoder_keys}
    mano_l1 = []

    with torch.no_grad():
        for batch in val_loader:
            mano = batch['mano_pose'].to(device)
            targets = batch['targets']
            B, T, Dm = mano.shape

            x_bt = mano.reshape(B * T, Dm)

            # 与训练相同路径：encode -> quantize -> 各 decoder
            _, preds, _ = model(x_bt, return_all=True)
            ys = resolve_targets(targets, args.decoder_keys, device)

            if hasattr(model, 'mano_decoder') and model.mano_decoder is not None:
                z_e = model.encode(x_bt)
                _, z_q, _, _, _ = model.quantize(z_e)
                mano_rec = model.mano_decoder(z_q).squeeze(-1)
                mano_l1.append(float(torch.nn.functional.l1_loss(mano_rec, x_bt).item()))

            for i, y in enumerate(ys):
                if i >= len(preds):
                    continue
                if y is None:
                    continue

                pred_seq = preds[i].view(B, T, -1)[0].cpu().numpy()
                gt_seq = y[0].cpu().numpy()
                rkey = args.decoder_keys[i]

                per_robot[rkey]["mpjpe_legacy"].append(mpjpe(pred_seq, gt_seq))
                per_robot[rkey]["qpos_mae"].append(float(np.abs(pred_seq[:, 6:] - gt_seq[:, 6:]).mean()))
                per_robot[rkey]["trans_mae"].append(fhlt(pred_seq, gt_seq))
                per_robot[rkey]["rot_mae"].append(fhlr(pred_seq, gt_seq))
                per_robot[rkey]["fid"].append(fid(pred_seq, gt_seq))
                per_robot[rkey]["smooth"].append(smoothness_l2(pred_seq))
                per_robot[rkey]["drift"].append(rollout_drift(pred_seq, gt_seq))

    print("===== Original UniHM VQVAE Eval =====")
    if len(mano_l1) > 0:
        print(f"mano_l1: {float(np.mean(mano_l1)):.6f}")

    macro = {m: [] for m in metric_names}
    for rkey in args.decoder_keys:
        vals = per_robot[rkey]
        if sum(len(vals[m]) for m in metric_names) == 0:
            continue
        print(f"\n[{rkey}]")
        for m in metric_names:
            mv = float(np.mean(vals[m])) if len(vals[m]) > 0 else float('nan')
            print(f"{m}: {mv:.6f}")
            if np.isfinite(mv):
                macro[m].append(mv)

    print("\n[macro_avg]")
    for m in metric_names:
        print(f"{m}: {float(np.mean(macro[m])) if len(macro[m]) > 0 else float('nan'):.6f}")


if __name__ == '__main__':
    p = argparse.ArgumentParser('Evaluate original UniHM VQVAE with the same metrics')
    p.add_argument('--seq-glob', type=str, default='/public/home/jiaozixun/UniHM/processed_dexycb/*.npz')
    p.add_argument('--ckpt', type=str, default='/public/home/jiaozixun/UniHM/paper_ckp/vqvae_with_mano_decoder_mano.pth')
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--num-workers', type=int, default=4)

    # 以下默认值与现有训练/推理脚本一致
    p.add_argument('--in-dim', type=int, default=1)
    p.add_argument('--h-dim', type=int, default=128)
    p.add_argument('--res-h-dim', type=int, default=128)
    p.add_argument('--n-res-layers', type=int, default=2)
    p.add_argument('--n-embeddings', type=int, default=8192)
    p.add_argument('--embedding-dim', type=int, default=512)
    p.add_argument('--beta', type=float, default=0.25)
    p.add_argument('--use-mlp', action='store_true', default=False)
    p.add_argument('--input-length', type=int, default=51)
    p.add_argument('--mano-dim', type=int, default=51)

    p.add_argument('--decoder-out-dims', type=str, default='22,30,26,22,16,8,18')
    p.add_argument(
        '--decoder-keys',
        type=str,
        default=','.join([
            'allegro_hand_qpos',
            'shadow_hand_qpos',
            'svh_hand_qpos',
            'leap_hand_qpos',
            'ability_hand_qpos',
            'panda_hand_qpos',
            'inspire_hand_qpos',
        ]),
    )

    args = p.parse_args()

    # 兼容：若用户只给了部分 keys，就按给定顺序评估；
    # 若未给，采用 SFT 中统一顺序的前 N 个。
    if not args.decoder_keys:
        args.decoder_keys = ','.join(ROBOT_KEYS_ORDER[:len(parse_int_list(args.decoder_out_dims))])

    evaluate(args)
