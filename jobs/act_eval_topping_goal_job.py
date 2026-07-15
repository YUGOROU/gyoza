# /// script
# requires-python = ">=3.11"
# dependencies = ["mujoco==3.10.0", "numpy", "lerobot==0.4.4", "torch", "torchvision", "imageio[ffmpeg]", "huggingface_hub"]
# ///
"""トッピング用ゴール条件付き ACT の sim 内評価（naruto/ajitama・丼シーン）。

act_eval_goal_job.py（球・平皿）のトッピング版。差分:
  1. 環境は GYOZA_TOMATO/GYOZA_BOWL env var で PickPlaceEnv が丼＋トッピングを構築
  2. TASK 文字列を datagen（act_datagen_topping_goal_job）と一致させる
  3. 静止判定を topping 対応（並進速度のみ ‖cvel[3:]‖）— 薄い円盤の回転で偽陰性になる罠を回避
  4. ゴール円盤は datagen と同一（中心 (0.204,-0.096)、半径 0.018）

事前宣言ゴール方式（relabeling 禁止）: 成功 = 終端で d_goal<THRESH かつ z ゾーン内・静定。

実行（naruto の例）:
    hf jobs uv run --flavor t4-small --timeout 2h --secrets HF_TOKEN \
        -v hf://buckets/YUGOROU/gyoza-sim:/gyoza --env GYOZA_DATA=/gyoza \
        --env GYOZA_TOMATO=naruto --env GYOZA_BOWL=0.204,-0.098 \
        --env POLICY=YUGOROU/act_gyoza_naruto_goal_synth \
        --env RUN=eval_act_naruto --env EPISODES=50 --env SEED=42 \
        jobs/act_eval_topping_goal_job.py

env vars: POLICY / RUN / EPISODES / VIDEO_N(5) / SEED / MAX_STEPS(450) / THRESH(0.025)
          / DISK_CX(0.204) / DISK_CY(-0.096) / OFFSET_R(0.018) / GYOZA_TOMATO / GYOZA_BOWL
"""

import json
import os
import pathlib
import subprocess
import sys

DATA = pathlib.Path(os.environ.get("GYOZA_DATA", "/gyoza"))
CODE = DATA / "code"


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

    topping = os.environ.get("GYOZA_TOMATO", "naruto")
    TASK = f"pick the {topping} and place it at the goal point in the ramen bowl"
    policy_repo = os.environ["POLICY"]
    run = os.environ.get("RUN", f"eval_act_{topping}")
    episodes = int(os.environ.get("EPISODES", "50"))
    video_n = int(os.environ.get("VIDEO_N", "5"))
    seed = int(os.environ.get("SEED", "42"))
    max_steps = int(os.environ.get("MAX_STEPS", "450"))
    thresh = float(os.environ.get("THRESH", "0.025"))
    disk_c = np.array([float(os.environ.get("DISK_CX", "0.204")),
                       float(os.environ.get("DISK_CY", "-0.096"))])
    offset_r = float(os.environ.get("OFFSET_R", "0.018"))
    out = DATA / "outputs" / "evals" / run
    out.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy = ACTPolicy.from_pretrained(policy_repo)
    policy.to(device).eval()
    cfg = PreTrainedConfig.from_pretrained(policy_repo)
    preprocessor, postprocessor = make_pre_post_processors(cfg, pretrained_path=policy_repo)
    # GYOZA_TOMATO/GYOZA_BOWL は PickPlaceEnv が os.environ から拾う（丼＋トッピング）
    env = PickPlaceEnv(seed=seed, render_obs=True)
    rng = np.random.default_rng(seed + 1)
    print(f"[eval-topping] topping={topping} policy={policy_repo} episodes={episodes} "
          f"thresh={thresh} bowl={os.environ.get('GYOZA_BOWL')}", flush=True)

    stats = []
    for ep in range(episodes):
        env.reset()
        policy.reset()
        theta = rng.uniform(0, 2 * np.pi)
        rad = offset_r * np.sqrt(rng.uniform(0, 1))
        goal = (disk_c + rad * np.array([np.cos(theta), np.sin(theta)])).astype(np.float32)
        goal_t = torch.from_numpy(goal).unsqueeze(0).to(device)
        frames = []
        for t in range(max_steps):
            img = env.render("overhead")
            if ep < video_n:
                frames.append(img.copy())
            batch = {
                "observation.images.overhead":
                    torch.from_numpy(img).permute(2, 0, 1).float().div(255).unsqueeze(0).to(device),
                "observation.state":
                    torch.from_numpy(env.state_deg().astype(np.float32)).unsqueeze(0).to(device),
                "observation.environment_state": goal_t,
                "task": [TASK],
            }
            batch = preprocessor(batch)
            with torch.inference_mode():
                action = policy.select_action(batch)
            action = postprocessor(action)
            env.step(action.squeeze(0).cpu().numpy().astype(np.float64))

        tom = env.tomato_pos()
        d_goal = float(np.linalg.norm(tom[:2] - goal))
        # topping 対応: 並進速度のみで静止判定（薄い円盤の回転で偽陰性を避ける）
        vel = float(np.linalg.norm(env.data.cvel[env._tomato_body][3:]))
        in_z = bool(env.zone_z[0] < tom[2] < env.zone_z[1])
        succ = bool(d_goal < thresh and in_z and vel < 0.05)
        rec = dict(episode=ep, success=succ, d_goal=round(d_goal, 4),
                   goal=[round(float(v), 4) for v in goal], settled=bool(vel < 0.05),
                   z=round(float(tom[2]), 4))
        stats.append(rec)
        print(f"[ep {ep:03d}] {rec}", flush=True)
        if frames:
            imageio.mimsave(out / f"ep{ep:03d}.mp4", frames, fps=30)

    d = [s["d_goal"] for s in stats]
    summary = dict(topping=topping, policy=policy_repo, episodes=episodes, seed=seed, thresh=thresh,
                   success_rate=sum(s["success"] for s in stats) / episodes,
                   d_goal_mean=round(float(np.mean(d)), 4),
                   d_goal_median=round(float(np.median(d)), 4),
                   per_episode=stats)
    (out / "stats.json").write_text(json.dumps(summary, indent=2))
    print("[eval-topping] DONE", json.dumps({k: v for k, v in summary.items() if k != "per_episode"}), flush=True)


if __name__ == "__main__":
    main()
