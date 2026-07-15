# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "mujoco",
#     "numpy",
#     "imageio[ffmpeg]",
#     "pillow",
#     "huggingface_hub",
# ]
# ///
"""VLA ゼロショット pick_place 成功率測定を HF Jobs 上で回すランチャ（実験計画 §4 手順1）。

bucket YUGOROU/gyoza-sim を /gyoza に rw マウントして使う（/data は予約済で不可）:
  - コード/アセット: /gyoza/code（ローカルから `hf buckets sync` 済みのもの）
  - 出力:            /gyoza/outputs/zeroshot/<RUN_NAME>/（ジョブが直接書き込む）

モデル系統の依存はインライン metadata に置かず `--with` で渡す（lerobot 版スキュー対策）:

  smolvla（t4-small で可）:
    hf jobs uv run --flavor t4-small --timeout 4h --secrets HF_TOKEN \
      -v hf://buckets/YUGOROU/gyoza-sim:/gyoza --env GYOZA_DATA=/gyoza \
      --with "lerobot[smolvla]" --with num2words \
      --env MODEL=smolvla --env EPISODES=50 --env SEED=0 jobs/zeroshot_job.py

  pi05（py3.12 + lerobot git main 必須 = VLA-Bench 知見。bf16 GPU 推奨 → l4x1）:
    hf jobs uv run --flavor l4x1 --timeout 4h --secrets HF_TOKEN --python 3.12 \
      -v hf://buckets/YUGOROU/gyoza-sim:/gyoza --env GYOZA_DATA=/gyoza \
      --with "lerobot[pi] @ git+https://github.com/huggingface/lerobot.git" \
      --env MODEL=pi05 --env EPISODES=50 --env SEED=0 jobs/zeroshot_job.py

  molmoact2（bf16 で <16GB だが余裕を見て l4x1。trust_remote_code の import 全列挙が必須:
             einops/torchvision/requests を欠くと processor ロードで ImportError）:
    hf jobs uv run --flavor l4x1 --timeout 4h --secrets HF_TOKEN \
      -v hf://buckets/YUGOROU/gyoza-sim:/gyoza --env GYOZA_DATA=/gyoza \
      --with torch --with transformers --with einops --with torchvision --with requests \
      --env MODEL=molmoact2 --env EPISODES=50 --env SEED=0 jobs/zeroshot_job.py

環境変数: MODEL / EPISODES / MAX_STEPS / SEED / VIDEO_EVERY / STATS_REPO / RUN_NAME
"""

import os
import pathlib
import subprocess
import sys

DATA = pathlib.Path(os.environ.get("GYOZA_DATA", "/data"))
CODE = DATA / "code"

MODEL = os.environ.get("MODEL", "smolvla")
EPISODES = os.environ.get("EPISODES", "50")
MAX_STEPS = os.environ.get("MAX_STEPS", "300")
SEED = os.environ.get("SEED", "0")
VIDEO_EVERY = os.environ.get("VIDEO_EVERY", "10")
STATS_REPO = os.environ.get("STATS_REPO", "YUGOROU/act_grasp_almond")
RUN_NAME = os.environ.get("RUN_NAME", f"{MODEL}_seed{SEED}")


def sh(cmd: str) -> str:
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return (r.stdout + r.stderr).strip()


def try_backend(backend: str) -> bool:
    code = (
        "import mujoco;"
        "m = mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom size=\"0.1\"/></worldbody></mujoco>');"
        "d = mujoco.MjData(m); mujoco.mj_forward(m, d);"
        "r = mujoco.Renderer(m, 240, 320); r.update_scene(d); r.render()"
    )
    env = dict(os.environ, MUJOCO_GL=backend)
    return subprocess.run([sys.executable, "-c", code], env=env,
                          capture_output=True, timeout=120).returncode == 0


def main():
    print("[info]", sh("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader") or "no GPU")
    assert CODE.is_dir(), f"{CODE} が無い（bucket を -v hf://buckets/YUGOROU/gyoza-sim:/gyoza でマウントし GYOZA_DATA=/gyoza を渡したか?）"
    sh("apt-get update -qq && apt-get install -y -qq libegl1 libgles2 libosmesa6 ffmpeg")

    backend = next((b for b in ("egl", "osmesa") if try_backend(b)), None)
    assert backend, "MuJoCo ヘッドレス描画がどのバックエンドでも不可"
    os.environ["MUJOCO_GL"] = backend
    print(f"[gl] MUJOCO_GL={backend}")

    out_dir = DATA / "outputs" / "zeroshot"
    sys.path.insert(0, str(CODE))
    sys.path.insert(0, str(CODE / "scripts"))
    sys.argv = [
        "zeroshot_eval.py",
        "--model", MODEL, "--episodes", EPISODES, "--max-steps", MAX_STEPS,
        "--seed", SEED, "--video-every", VIDEO_EVERY, "--stats-repo", STATS_REPO,
        "--out", str(out_dir),
    ]
    import zeroshot_eval
    zeroshot_eval.main()

    # zeroshot_eval は <out>/<model>_seed<seed>/ に書く。RUN_NAME が別名ならリネーム
    default_dir = out_dir / f"{MODEL}_seed{SEED}"
    if RUN_NAME != f"{MODEL}_seed{SEED}":
        default_dir.rename(out_dir / RUN_NAME)
    print(f"[done] results at bucket gyoza-sim: outputs/zeroshot/{RUN_NAME}")


if __name__ == "__main__":
    main()
