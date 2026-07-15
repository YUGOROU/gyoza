# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "mujoco",
#     "numpy",
#     "gymnasium",
#     "stable-baselines3",
#     "tensorboard",
#     "huggingface_hub",
# ]
# ///
"""RL エキスパート学習を HF Jobs CPU で回すランチャ（報酬バリアント並列スイープ用）。

描画なし（render_obs=False）なので GPU 不要。cpu-upgrade（8 vCPU, $0.03/h）で十分。

実行例（バリアント v2b・10M ステップ）:
    hf jobs uv run --flavor cpu-upgrade --timeout 8h --secrets HF_TOKEN \
        -v hf://buckets/YUGOROU/gyoza-sim:/gyoza --env GYOZA_DATA=/gyoza \
        --env REWARD_VARIANT=v2b --env STEPS=10000000 jobs/rl_job.py

環境変数: REWARD_VARIANT (v2a/v2b/v2c) / STEPS / N_ENVS / SEED
出力: bucket outputs/rl/<REWARD_VARIANT>_seed<SEED>/
"""

import os
import pathlib
import sys

DATA = pathlib.Path(os.environ.get("GYOZA_DATA", "/gyoza"))
CODE = DATA / "code"

VARIANT = os.environ.get("REWARD_VARIANT", "v2a")
STEPS = os.environ.get("STEPS", "10000000")
N_ENVS = os.environ.get("N_ENVS", "8")
SEED = os.environ.get("SEED", "0")


def main():
    assert CODE.is_dir(), f"{CODE} が無い（bucket マウント + GYOZA_DATA を確認）"
    sys.path.insert(0, str(CODE))
    sys.path.insert(0, str(CODE / "scripts"))
    out = DATA / "outputs" / "rl" / f"{VARIANT}_seed{SEED}"
    sys.argv = ["rl_train.py", "--steps", STEPS, "--n-envs", N_ENVS,
                "--seed", SEED, "--out", str(out)]
    print(f"[rl_job] variant={VARIANT} steps={STEPS} n_envs={N_ENVS} → {out}")
    import rl_train
    rl_train.main()


if __name__ == "__main__":
    main()
