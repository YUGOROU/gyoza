# /// script
# requires-python = ">=3.11"
# dependencies = ["mujoco", "numpy", "gymnasium", "stable-baselines3", "torch", "huggingface_hub", "lerobot==0.4.4"]
# ///
"""ハイブリッド expert → LeRobot データセット直接書き出しジョブ（ACT 蒸留用）。

rollout_job.py の FINISH=script ロジックを流用し、全ステップで俯瞰カメラを描画。
成功エピソード（success_final）のみ LeRobotDataset に追加し、HF Hub へ push する。
（旧 datagen_s* の npz は先頭3ep しか動画がなく ACT 学習に使えないため再生成）

(s_t, a_t) 規約: 観測（画像+状態deg）はアクション適用前にキャプチャ。
action = 6次元 absolute 関節目標（度）@30Hz（アダプタ境界の度契約と一致）。

実行:
    hf jobs uv run --flavor t4-small --timeout 3h --secrets HF_TOKEN \
        -v hf://buckets/YUGOROU/gyoza-sim:/gyoza --env GYOZA_DATA=/gyoza \
        --env MODEL=outputs/rl_v3a-warm-perf/best/best_model.zip \
        --env REPO_ID=YUGOROU/gyoza-pickplace-synth \
        --env SEEDS=100,200,300 --env EPISODES=100 \
        jobs/act_datagen_job.py

env vars: MODEL(bucket 相対) / REPO_ID / SEEDS(カンマ区切り) / EPISODES(シード毎)

層化サンプリングモード（2026-07-11 追加。奥側デッドゾーン=棄却バイアスの根治）:
    CELLS="0,1,2,3" を渡すと SEEDS/EPISODES の代わりに、ライブ領域の 4×4 グリッド
    （cell = ix*4+iy、x: TOMATO_X を4分割 / y: TOMATO_Y を4分割）の各セルについて
    「keep が KEEP_PER_CELL(既定14) に達するまで最大 MAX_ATT_PER_CELL(既定40) 試行」する。
    初期位置はセル内に入るまで rejection reset（テレポートなし = obs 整合を保つ）。
    SEED(既定77) はセル毎の env シード基底。
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

    import numpy as np
    from stable_baselines3 import PPO

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from gyoza.envs.pick_place import JOINTS
    from gyoza.envs.rl_pick_place import RLPickPlaceEnv

    model_rel = os.environ["MODEL"]
    repo_id = os.environ["REPO_ID"]
    seeds = [int(s) for s in os.environ.get("SEEDS", "100,200,300").split(",")]
    episodes = int(os.environ.get("EPISODES", "100"))

    features = {
        "observation.images.overhead": {"dtype": "video", "shape": (480, 640, 3),
                                        "names": ["height", "width", "channels"]},
        "observation.state": {"dtype": "float32", "shape": (6,), "names": JOINTS},
        "action": {"dtype": "float32", "shape": (6,), "names": JOINTS},
    }
    ds = LeRobotDataset.create(repo_id, fps=30, features=features,
                               root="/tmp/lerobot_ds", robot_type="so101")

    model = PPO.load(str(DATA / model_rel), device="cpu")
    GRIP_OPEN = 40.0
    FROZEN_GOAL = np.array([0.26, -0.07, 0.014])  # 学習時の皿 site 位置（obs 凍結）
    kept, total = 0, 0

    def run_episode(env, obs):
        """1 エピソード実行。(succ, buf) を返す。buf = [(frame, state_deg, action_deg)]"""
        buf = []

        def record_then(target_deg, frame, state):
            buf.append((frame, state, np.asarray(target_deg, dtype=np.float32).copy()))

        # phase 1: RL ポリシー（静定把持ホバーで打ち切り）
        hover = 0
        prev_tom = env.env.tomato_pos()
        for t in range(250):
            frame = env.env.render("overhead").copy()
            state = env.env.state_deg().astype(np.float32).copy()
            a, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(a)
            record_then(np.rad2deg(env._target), frame, state)
            tom = env.env.tomato_pos()
            settled = float(np.linalg.norm(tom - prev_tom)) < 0.004
            prev_tom = tom
            hover = hover + 1 if (info["grasped"] and info["d_place"] < 0.12 and settled) else 0
            if trunc or hover >= 20:
                break

        # phase 2: グリッパー開放 → 退避
        succ = False
        if hover >= 20:
            tgt = np.rad2deg(env._target).astype(np.float64).copy()
            for k in range(1, 16):
                tgt[5] = -10.0 + (GRIP_OPEN + 10.0) * (k / 15)
                frame = env.env.render("overhead").copy()
                state = env.env.state_deg().astype(np.float32).copy()
                env.env.step(tgt)
                record_then(tgt, frame, state)
            tgt[5] = GRIP_OPEN
            tgt[1] -= 25.0
            for _ in range(45):
                frame = env.env.render("overhead").copy()
                state = env.env.state_deg().astype(np.float32).copy()
                env.env.step(tgt)
                record_then(tgt, frame, state)
                if env.env.check_success():
                    break
            succ = env.env.check_success()
        return succ, buf

    def keep_episode(buf):
        for frame, state, action in buf:
            ds.add_frame({"observation.images.overhead": frame,
                          "observation.state": state,
                          "action": action,
                          "task": TASK})
        ds.save_episode()

    cells_env = os.environ.get("CELLS", "")
    if cells_env:
        # 層化モード: セル毎に keep が目標に達するまで rejection reset で再試行
        from gyoza.envs.pick_place import TOMATO_X, TOMATO_Y
        keep_per_cell = int(os.environ.get("KEEP_PER_CELL", "14"))
        max_att = int(os.environ.get("MAX_ATT_PER_CELL", "40"))
        base_seed = int(os.environ.get("SEED", "77"))
        cell_stats = {}
        for c in [int(x) for x in cells_env.split(",")]:
            ix, iy = c // 4, c % 4
            x0 = TOMATO_X[0] + ix * (TOMATO_X[1] - TOMATO_X[0]) / 4
            y0 = TOMATO_Y[0] + iy * (TOMATO_Y[1] - TOMATO_Y[0]) / 4
            x1 = x0 + (TOMATO_X[1] - TOMATO_X[0]) / 4
            y1 = y0 + (TOMATO_Y[1] - TOMATO_Y[0]) / 4
            env = RLPickPlaceEnv(seed=base_seed * 1000 + c, render_obs=True)
            env.env.plate_pos = lambda: FROZEN_GOAL.copy()
            keeps_c = 0
            for att in range(max_att):
                if keeps_c >= keep_per_cell:
                    break
                for _ in range(300):  # セル内に入るまで reset（描画なし・安価）
                    obs, _ = env.reset()
                    px, py = env.env.tomato_pos()[:2]
                    if x0 <= px < x1 and y0 <= py < y1:
                        break
                succ, buf = run_episode(env, obs)
                total += 1
                print(f"[cell {c} att {att:02d}] success={succ} steps={len(buf)}", flush=True)
                if succ:
                    keep_episode(buf)
                    kept += 1
                    keeps_c += 1
            cell_stats[c] = keeps_c
            print(f"[cell {c}] keeps={keeps_c}/{keep_per_cell}", flush=True)
        print("[datagen] cell_stats", json.dumps(cell_stats), flush=True)
    else:
        for seed in seeds:
            env = RLPickPlaceEnv(seed=seed, render_obs=True)
            env.env.plate_pos = lambda: FROZEN_GOAL.copy()
            for ep in range(episodes):
                obs, _ = env.reset()
                succ, buf = run_episode(env, obs)
                total += 1
                print(f"[seed {seed} ep {ep:03d}] success={succ} steps={len(buf)}", flush=True)
                if succ:
                    keep_episode(buf)
                    kept += 1

    print(f"[datagen] kept {kept}/{total} episodes -> push {repo_id}", flush=True)
    if kept > 0:
        ds.finalize()  # 必須: parquet writer を close(欠くとフッターなしの壊れた parquet が push される)
        ds.push_to_hub(private=True)
    print("[datagen] DONE", json.dumps({"kept": kept, "total": total, "repo_id": repo_id}), flush=True)


if __name__ == "__main__":
    main()
