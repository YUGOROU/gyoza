# /// script
# requires-python = ">=3.11"
# dependencies = ["mujoco", "numpy", "gymnasium", "stable-baselines3", "tensorboard", "huggingface_hub"]
# ///
"""RL エキスパート学習（PPO・状態ベース）を HF Jobs CPU で実行するランチャ。

描画なし（render_obs=False）の物理オンリーなので CPU flavor で十分。
出力は bucket の outputs/rl_<RUN> に直接書き込む。

実行:
    hf jobs uv run --flavor cpu-upgrade --timeout 3h --secrets HF_TOKEN \
        -v hf://buckets/YUGOROU/gyoza-sim:/gyoza --env GYOZA_DATA=/gyoza \
        --env RUN=v2a --env REWARD_VARIANT=v2a --env STEPS=3000000 --env N_ENVS=8 \
        --env OMP_NUM_THREADS=1 --env MKL_NUM_THREADS=1 --env OPENBLAS_NUM_THREADS=1 \
        jobs/rl_train_job.py

注意: スレッド制限 env var は必須。64 コアマシンで各サブプロセスの BLAS/torch が
全コア分スレッドを立てて競合し、~690 fps まで劣化する（制限すると ~2,700-3,800 fps）。

env vars: RUN(出力サフィックス) / REWARD_VARIANT(v2a|v2b|v2c) / STEPS / N_ENVS / SEED
         / INIT(bucket 内の PPO zip 相対パス、warm-start 用。例 outputs/rl_v2a/final.zip)
"""

import os
import pathlib
import subprocess
import sys

DATA = pathlib.Path(os.environ.get("GYOZA_DATA", "/gyoza"))
CODE = DATA / "code"

run = os.environ.get("RUN", "v2a")
steps = os.environ.get("STEPS", "3000000")
n_envs = os.environ.get("N_ENVS", "8")
seed = os.environ.get("SEED", "0")
out = DATA / "outputs" / f"rl_{run}"

assert CODE.is_dir(), f"{CODE} が無い（bucket マウントと GYOZA_DATA を確認）"
print(f"[rl-job] run={run} variant={os.environ.get('REWARD_VARIANT', 'v2a')} "
      f"steps={steps} n_envs={n_envs} seed={seed} out={out}", flush=True)
print("[rl-job] cpu:", os.cpu_count(), flush=True)

cmd = [sys.executable, str(CODE / "scripts" / "rl_train.py"),
       "--steps", steps, "--n-envs", n_envs, "--seed", seed, "--out", str(out)]
if os.environ.get("INIT"):
    cmd += ["--init", str(DATA / os.environ["INIT"])]
r = subprocess.run(cmd, env=dict(os.environ, PYTHONPATH=str(CODE)))
sys.exit(r.returncode)
