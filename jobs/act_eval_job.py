# /// script
# requires-python = ">=3.11"
# dependencies = ["mujoco", "numpy", "lerobot==0.4.4", "torch", "torchvision", "imageio[ffmpeg]", "huggingface_hub"]
# ///
"""学習済み ACT ポリシーの sim 内評価ジョブ（新レイアウト・キッチンシーン）。

HF Hub から ACTPolicy をロードし、閉ループで N エピソード実行。
成功 = GT 事後条件 check_success()（トマトが皿ゾーン内 + 静止）。
先頭 VIDEO_N エピソードは俯瞰 mp4 を bucket に保存。

注意: 既存 zeroshot 測定（全 0%）は旧レイアウト。ACT 評価はこの新レイアウトで統一。

実行:
    hf jobs uv run --flavor t4-small --timeout 2h --secrets HF_TOKEN \
        -v hf://buckets/YUGOROU/gyoza-sim:/gyoza --env GYOZA_DATA=/gyoza \
        --env POLICY=YUGOROU/act_gyoza_pickplace_synth \
        --env RUN=eval_act_synth --env EPISODES=50 --env SEED=42 \
        jobs/act_eval_job.py

env vars: POLICY(hub repo) / RUN / EPISODES / VIDEO_N(既定5) / SEED / MAX_STEPS(既定400)
"""

import json
import os
import pathlib
import subprocess
import sys

DATA = pathlib.Path(os.environ.get("GYOZA_DATA", "/gyoza"))
CODE = DATA / "code"

TASK = "pick the tomato and place it on the plate"


def pick_gl_backend() -> str:
    probe = (
        "import mujoco;"
        "m = mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom size=\"0.1\"/></worldbody></mujoco>');"
        "d = mujoco.MjData(m); mujoco.mj_forward(m, d);"
        "mujoco.Renderer(m, 64, 64).update_scene(d)"
    )
    for backend in ("egl", "osmesa"):
        r = subprocess.run([sys.executable, "-c", probe],
                           env=dict(os.environ, MUJOCO_GL=backend),
                           capture_output=True, text=True, timeout=120)
        print(f"[gl] {backend}: {'OK' if r.returncode == 0 else 'NG'}", flush=True)
        if r.returncode == 0:
            return backend
    raise RuntimeError("描画バックエンドなし")


def main():
    subprocess.run("apt-get update -qq && apt-get install -y -qq libegl1 libgles2 libosmesa6 ffmpeg",
                   shell=True, capture_output=True)
    os.environ["MUJOCO_GL"] = pick_gl_backend()
    sys.path.insert(0, str(CODE))

    import imageio
    import numpy as np
    import torch

    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors
    from gyoza.envs.pick_place import PickPlaceEnv

    policy_repo = os.environ["POLICY"]
    run = os.environ.get("RUN", "eval_act")
    episodes = int(os.environ.get("EPISODES", "50"))
    video_n = int(os.environ.get("VIDEO_N", "5"))
    seed = int(os.environ.get("SEED", "42"))
    max_steps = int(os.environ.get("MAX_STEPS", "400"))
    out = DATA / "outputs" / "evals" / run
    out.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy = ACTPolicy.from_pretrained(policy_repo)
    policy.to(device).eval()
    # 重要: lerobot 0.4.4 は正規化をモデル外の processor パイプラインとして保存する。
    # これを通さないと生の度数入力 + 正規化空間の出力になり、定数アクション（平均値）
    # に崩壊する（2026-07-07 の 0% はこれが原因）
    cfg = PreTrainedConfig.from_pretrained(policy_repo)
    preprocessor, postprocessor = make_pre_post_processors(cfg, pretrained_path=policy_repo)
    env = PickPlaceEnv(seed=seed, render_obs=True)
    print(f"[eval] policy={policy_repo} episodes={episodes} device={device}", flush=True)

    stats = []
    for ep in range(episodes):
        env.reset()
        policy.reset()
        frames = []
        succ = False
        for t in range(max_steps):
            img = env.render("overhead")
            if ep < video_n:
                frames.append(img.copy())
            batch = {
                "observation.images.overhead":
                    torch.from_numpy(img).permute(2, 0, 1).float().div(255).unsqueeze(0).to(device),
                "observation.state":
                    torch.from_numpy(env.state_deg().astype(np.float32)).unsqueeze(0).to(device),
                "task": [TASK],
            }
            batch = preprocessor(batch)
            with torch.inference_mode():
                action = policy.select_action(batch)
            action = postprocessor(action)
            env.step(action.squeeze(0).cpu().numpy().astype(np.float64))
            if env.check_success():
                succ = True
                break
        rec = dict(episode=ep, success=bool(succ), steps=t + 1)
        stats.append(rec)
        print(f"[ep {ep:03d}] {rec}", flush=True)
        if frames:
            imageio.mimsave(out / f"ep{ep:03d}.mp4", frames, fps=30)

    summary = dict(policy=policy_repo, episodes=episodes, seed=seed,
                   success_rate=sum(s["success"] for s in stats) / episodes,
                   per_episode=stats)
    # bucket FUSE は空ディレクトリを実体化しない（VIDEO_N=0 だと冒頭 mkdir が消える）
    out.mkdir(parents=True, exist_ok=True)
    (out / "stats.json").write_text(json.dumps(summary, indent=2))
    print("[eval] DONE", json.dumps({k: v for k, v in summary.items() if k != "per_episode"}), flush=True)


if __name__ == "__main__":
    main()
