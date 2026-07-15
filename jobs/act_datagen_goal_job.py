# /// script
# requires-python = ">=3.11"
# dependencies = ["mujoco", "numpy", "gymnasium", "stable-baselines3", "torch", "huggingface_hub", "lerobot==0.4.4"]
# ///
"""ゴール条件付き ACT 用データ生成ジョブ（並列B 本命）。

ハイブリッド expert + ラストインチ制御（スパイク v6 確定版）で、
「皿上の指定点へ置く」エピソードを生成し LeRobotDataset として HF Hub へ push。

確定設計（docs/HANDOFF.md 2026-07-07）:
- ゴール円盤: 実測ホバー到達点 + (0, +DISK_Y) を中心に半径 OFFSET_R。
  スパイク v5/v6 の知見: 奥側(+y)は降下が完走する（手前側は elbow 可動域の谷で停滞）。
  中心+2.2cm/半径1.8cm なら境界(y≈0)を踏まず、かつ皿(半径5.5cm)内に全ゴールが収まる
- 指令は事前・ラベルは事後: 静定後の実測トマト xy をエピソード全フレームの
  observation.environment_state として遡及書き込み（hindsight relabeling）
- 棄却: ホバー未達 / 移動・降下中の落下 / check_success() 不成立（ゾーン外・非静定）
- 評価側は relabeling 禁止（事前宣言ゴールとの距離、閾値 2.5cm 提案）

実行:
    hf jobs uv run --flavor t4-small --timeout 4h --secrets HF_TOKEN \
        -v hf://buckets/YUGOROU/gyoza-sim:/gyoza --env GYOZA_DATA=/gyoza \
        --env MODEL=outputs/rl_v3a-warm-perf/best/best_model.zip \
        --env REPO_ID=YUGOROU/gyoza-pickplace-goal-synth \
        --env SEEDS=100,200,300 --env EPISODES=100 \
        jobs/act_datagen_goal_job.py

env vars: MODEL / REPO_ID / SEEDS / EPISODES(シード毎) / OFFSET_R(0.018) / DISK_Y(0.022)
"""

import json
import os
import pathlib
import subprocess
import sys

DATA = pathlib.Path(os.environ.get("GYOZA_DATA", "/gyoza"))
CODE = DATA / "code"

TASK = "pick the tomato and place it at the goal point on the plate"


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

    import mujoco
    import numpy as np
    from stable_baselines3 import PPO

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from gyoza.envs.pick_place import JOINTS
    from gyoza.envs.rl_pick_place import RLPickPlaceEnv

    model_rel = os.environ["MODEL"]
    repo_id = os.environ["REPO_ID"]
    seeds = [int(s) for s in os.environ.get("SEEDS", "100,200,300").split(",")]
    episodes = int(os.environ.get("EPISODES", "100"))
    offset_r = float(os.environ.get("OFFSET_R", "0.018"))
    disk_y = float(os.environ.get("DISK_Y", "0.022"))

    MOVE_STEP = 0.003
    DQ_MAX = 0.02
    SETTLE_N = 5
    MOVE_TIMEOUT = 120
    GRIP_OPEN = 40.0
    FROZEN_GOAL = np.array([0.26, -0.07, 0.014])

    features = {
        "observation.images.overhead": {"dtype": "video", "shape": (480, 640, 3),
                                        "names": ["height", "width", "channels"]},
        "observation.state": {"dtype": "float32", "shape": (6,), "names": JOINTS},
        "observation.environment_state": {"dtype": "float32", "shape": (2,),
                                          "names": ["goal_x", "goal_y"]},
        "action": {"dtype": "float32", "shape": (6,), "names": JOINTS},
    }
    ds = LeRobotDataset.create(repo_id, fps=30, features=features,
                               root="/tmp/lerobot_ds", robot_type="so101")

    model = PPO.load(str(DATA / model_rel), device="cpu")
    kept, total, achieved_log = 0, 0, []

    for seed in seeds:
        env = RLPickPlaceEnv(seed=seed, render_obs=True)
        env.env.plate_pos = lambda: FROZEN_GOAL.copy()
        m, d = env.env.model, env.env.data
        arm_dofs = [m.joint(f"follower_{j}").dofadr[0] for j in JOINTS[:5]]
        tcp_site = env.env._tcp_site
        CTRL_LO, CTRL_HI = env._ctrl_lo.copy(), env._ctrl_hi.copy()
        rng = np.random.default_rng(seed + 7)
        jacp = np.zeros((3, m.nv))

        for ep in range(episodes):
            obs, _ = env.reset()
            buf = []  # (frame, state_deg, action_deg)

            def in_hand() -> bool:
                return float(np.linalg.norm(env.env.tomato_pos() - env.env.jaw_mid_pos())) < 0.05

            def do_step(target_rad):
                frame = env.env.render("overhead").copy()
                state = env.env.state_deg().astype(np.float32).copy()
                env.env.step(np.rad2deg(target_rad))
                buf.append((frame, state, np.rad2deg(target_rad).astype(np.float32)))

            # --- phase 1: RL で静定ホバーへ ---
            hover = 0
            prev_tom = env.env.tomato_pos()
            for t in range(250):
                frame = env.env.render("overhead").copy()
                state = env.env.state_deg().astype(np.float32).copy()
                a, _ = model.predict(obs, deterministic=True)
                obs, r, term, trunc, info = env.step(a)
                buf.append((frame, state, np.rad2deg(env._target).astype(np.float32)))
                tom = env.env.tomato_pos()
                settled = float(np.linalg.norm(tom - prev_tom)) < 0.004
                prev_tom = tom
                hover = hover + 1 if (info["grasped"] and info["d_place"] < 0.12 and settled) else 0
                if trunc or hover >= 20:
                    break
            total += 1
            if hover < 20:
                print(f"[seed {seed} ep {ep:03d}] reject: rl_fail", flush=True)
                continue

            # --- phase 2: 微分 IK でゴールへ（指令ゴール = 円盤サンプル）---
            anchor = env.env.tomato_pos()[:2].copy()
            theta = rng.uniform(0, 2 * np.pi)
            rad = offset_r * np.sqrt(rng.uniform(0, 1))
            goal = anchor + np.array([0.0, disk_y]) + rad * np.array([np.cos(theta), np.sin(theta)])
            z_hold = env.env.tcp_pos()[2]
            tgt = env._target.copy()

            dropped, ok, settle = False, False, 0
            for t in range(MOVE_TIMEOUT):
                tom = env.env.tomato_pos()
                e_xy = goal - tom[:2]
                if not in_hand():
                    dropped = True
                    break
                if np.linalg.norm(e_xy) < 0.005:
                    settle += 1
                    if settle >= SETTLE_N:
                        ok = True
                        break
                else:
                    settle = 0
                dx = np.zeros(3)
                n = np.linalg.norm(e_xy)
                dx[:2] = e_xy if n < MOVE_STEP else e_xy / n * MOVE_STEP
                dx[2] = np.clip(z_hold - env.env.tcp_pos()[2], -MOVE_STEP, MOVE_STEP)
                mujoco.mj_jacSite(m, d, jacp, None, tcp_site)
                dq = np.clip(np.linalg.pinv(jacp[:, arm_dofs], rcond=1e-4) @ dx, -DQ_MAX, DQ_MAX)
                tgt[:5] = np.clip(tgt[:5] + dq, CTRL_LO[:5], CTRL_HI[:5])
                do_step(tgt)
            if dropped or not ok:
                print(f"[seed {seed} ep {ep:03d}] reject: move dropped={dropped}", flush=True)
                continue

            # --- phase 2.5: 降下（停滞したらその高さで開放）---
            DROP_CLEAR = 0.004
            rest_z = 0.015 + 0.016
            z_hist = []
            for t in range(200):
                tom = env.env.tomato_pos()
                e_z = (rest_z + DROP_CLEAR) - tom[2]
                if not in_hand():
                    dropped = True
                    break
                if abs(e_z) < 0.002:
                    break
                z_hist.append(tom[2])
                if len(z_hist) > 25 and z_hist[-26] - tom[2] < 0.001:
                    break
                dx = np.zeros(3)
                dx[:2] = np.clip(goal - tom[:2], -MOVE_STEP, MOVE_STEP)
                dx[2] = np.clip(e_z, -MOVE_STEP, MOVE_STEP)
                mujoco.mj_jacSite(m, d, jacp, None, tcp_site)
                dq = np.clip(np.linalg.pinv(jacp[:, arm_dofs], rcond=1e-4) @ dx, -DQ_MAX, DQ_MAX)
                tgt[:5] = np.clip(tgt[:5] + dq, CTRL_LO[:5], CTRL_HI[:5])
                do_step(tgt)
            if dropped:
                print(f"[seed {seed} ep {ep:03d}] reject: descend drop", flush=True)
                continue

            # --- phase 3: 開放 → 真上へ抜く → 退避 ---
            for k in range(1, 16):
                tgt[5] = np.deg2rad(-10.0 + (GRIP_OPEN + 10.0) * (k / 15))
                do_step(tgt)
            z_up = env.env.tcp_pos()[2] + 0.03
            for t in range(25):
                dx = np.array([0.0, 0.0, np.clip(z_up - env.env.tcp_pos()[2], -MOVE_STEP, MOVE_STEP)])
                mujoco.mj_jacSite(m, d, jacp, None, tcp_site)
                dq = np.clip(np.linalg.pinv(jacp[:, arm_dofs], rcond=1e-4) @ dx, -DQ_MAX, DQ_MAX)
                tgt[:5] = np.clip(tgt[:5] + dq, CTRL_LO[:5], CTRL_HI[:5])
                do_step(tgt)
            tgt[1] -= np.deg2rad(25.0)
            for t in range(45):
                do_step(tgt)
                if env.env.check_success():
                    break

            if not env.env.check_success():
                print(f"[seed {seed} ep {ep:03d}] reject: not in zone/settled", flush=True)
                continue

            # --- hindsight relabeling: 実測静定点をゴールとして全フレームに書き込む ---
            achieved = env.env.tomato_pos()[:2].astype(np.float32).copy()
            achieved_log.append([round(float(v), 4) for v in (achieved - anchor)])
            for frame, state, action in buf:
                ds.add_frame({"observation.images.overhead": frame,
                              "observation.state": state,
                              "observation.environment_state": achieved,
                              "action": action,
                              "task": TASK})
            ds.save_episode()
            kept += 1
            print(f"[seed {seed} ep {ep:03d}] keep #{kept} achieved_off={achieved_log[-1]} "
                  f"cmd_err={float(np.linalg.norm(achieved - goal)):.4f} steps={len(buf)}", flush=True)

    print(f"[datagen-goal] kept {kept}/{total} -> push {repo_id}", flush=True)
    ds.finalize()  # 必須: parquet writer close（欠くと壊れた parquet が push される）
    ds.push_to_hub(private=True)
    out = DATA / "outputs" / "datagen_goal"
    out.mkdir(parents=True, exist_ok=True)
    (out / "achieved_hist.json").write_text(json.dumps(
        dict(kept=kept, total=total, achieved_offsets=achieved_log), indent=2))
    print("[datagen-goal] DONE", json.dumps({"kept": kept, "total": total, "repo_id": repo_id}), flush=True)


if __name__ == "__main__":
    main()
