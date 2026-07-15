# /// script
# requires-python = ">=3.11"
# dependencies = ["lerobot==0.4.4", "torch", "torchvision", "huggingface_hub"]
# ///
"""ACT 学習ジョブ（合成データ条件）。

HF Hub のデータセット（act_datagen_job.py が push したもの）から ACT を学習。
- 学習は job ローカルディスクで実行（bucket マウントへの torch.save は避ける）
- 完走時: 最終ポリシーを HF Hub へ push + checkpoints ディレクトリを bucket へコピー

実行（a100-large、100k steps）:
    hf jobs uv run --flavor a100-large --timeout 8h --secrets HF_TOKEN \
        -v hf://buckets/YUGOROU/gyoza-sim:/gyoza --env GYOZA_DATA=/gyoza \
        --env DATASET=YUGOROU/gyoza-pickplace-synth \
        --env POLICY_REPO=YUGOROU/act_gyoza_pickplace_synth \
        --env RUN=act_synth --env STEPS=100000 \
        jobs/act_train_job.py

env vars: DATASET / POLICY_REPO / RUN / STEPS / BATCH(既定8) / SEED(既定1000)
"""

import os
import pathlib
import shutil
import subprocess
import sys

DATA = pathlib.Path(os.environ.get("GYOZA_DATA", "/gyoza"))


def main():
    # torchcodec はシステムの FFmpeg 共有ライブラリ（libavutil 等）に動的リンクする。
    # 未インストールだと "Could not load libtorchcodec_core*.so" で学習開始時に落ちる
    subprocess.run("apt-get update -qq && apt-get install -y -qq ffmpeg",
                   shell=True, capture_output=True)
    dataset = os.environ["DATASET"]
    policy_repo = os.environ["POLICY_REPO"]
    run = os.environ.get("RUN", "act_synth")
    steps = os.environ.get("STEPS", "100000")
    batch = os.environ.get("BATCH", "8")
    seed = os.environ.get("SEED", "1000")
    out = pathlib.Path("/tmp/train") / run

    cmd = [
        sys.executable, "-m", "lerobot.scripts.lerobot_train",
        f"--dataset.repo_id={dataset}",
        "--policy.type=act",
        f"--policy.repo_id={policy_repo}",
        "--policy.push_to_hub=true",
        "--policy.device=cuda",
        f"--output_dir={out}",
        f"--job_name={run}",
        f"--steps={steps}",
        f"--batch_size={batch}",
        f"--seed={seed}",
        "--save_freq=20000",
        "--eval_freq=0",
        "--wandb.enable=false",
    ]
    print("[train]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)

    # checkpoints を bucket に退避（最新番号のみ保持方針だが、20k 刻み全部で ~数百MB なので全コピー）
    dst = DATA / "outputs" / run / "checkpoints"
    ckpts = out / "checkpoints"
    if ckpts.exists():
        shutil.copytree(ckpts, dst, dirs_exist_ok=True)
        print(f"[train] checkpoints copied -> {dst}", flush=True)
    print("[train] DONE", flush=True)


if __name__ == "__main__":
    main()
