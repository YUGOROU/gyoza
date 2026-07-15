"""RL エキスパート学習（PPO・状態ベース・ゴール条件付き）— RL スパイク本体。

ローカル M3 で回す想定（物理オンリーなので描画不要・CPU で十分）。
収束判定: eval の success_rate。>0.8 なら教師データ工場として合格、
1日回して立ち上がらなければテレオペ分岐（ユーザー決定 2026-07-05）。

実行:
    python scripts/rl_train.py --steps 3000000 --n-envs 8
    tensorboard --logdir outputs/rl/tb   # 任意
"""

from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))


def make_env(rank: int, seed: int):
    def _f():
        from stable_baselines3.common.monitor import Monitor
        from gyoza.envs.rl_pick_place import RLPickPlaceEnv
        return Monitor(RLPickPlaceEnv(seed=seed + rank), info_keywords=("is_success",))
    return _f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3_000_000)
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="outputs/rl")
    ap.add_argument("--init", default=None,
                    help="warm-start 用の既存 PPO zip（ポリシー/価値のパラメータのみ継承）")
    args = ap.parse_args()

    import os

    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
    from stable_baselines3.common.vec_env import SubprocVecEnv

    # env サブプロセスは OMP_NUM_THREADS=1 で絞りつつ、learner の GEMM には複数スレッドを許可
    torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "8")))

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    venv = SubprocVecEnv([make_env(i, args.seed) for i in range(args.n_envs)])
    eval_env = SubprocVecEnv([make_env(1000, args.seed)])

    model = PPO(
        "MlpPolicy", venv,
        n_steps=2048, batch_size=512, n_epochs=10,
        learning_rate=3e-4, gamma=0.99, gae_lambda=0.95,
        ent_coef=0.005, clip_range=0.2,
        policy_kwargs=dict(net_arch=[256, 256]),
        tensorboard_log=str(out / "tb"),
        seed=args.seed, verbose=1,
    )
    if args.init:
        model.set_parameters(args.init, device="cpu")
        print(f"[rl] warm-start from {args.init}")
    model.learn(
        total_timesteps=args.steps,
        callback=[
            EvalCallback(eval_env, best_model_save_path=str(out / "best"),
                         n_eval_episodes=20, eval_freq=50_000 // args.n_envs,
                         deterministic=True),
            CheckpointCallback(save_freq=500_000 // args.n_envs,
                               save_path=str(out / "ckpt"), name_prefix="ppo"),
        ],
        progress_bar=False,
    )
    model.save(out / "final")
    print(f"[rl] saved to {out}")


if __name__ == "__main__":
    main()
