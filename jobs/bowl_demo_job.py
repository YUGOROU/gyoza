# /// script
# requires-python = ">=3.11"
# dependencies = ["mujoco==3.10.0", "numpy", "gymnasium", "stable-baselines3", "torch", "imageio[ffmpeg]", "huggingface_hub"]
# ///
# 注: mujoco はローカル(.venv)と同一の 3.10.0 に固定（RL 閉ループ把持は版差でずれる。
#     lerobot 0.4.4 固定と同じ衛生。rollout_job.py と同方針）。
"""丼盛り付けデモ — lastinch 降下 + トッピング別開放を丼シーンで統合。

GYOZA 核ループ（VLM 事後条件判定 → retry → 次スキル）のデモ用フック。本ジョブは
その前提となる「単一トッピングを丼内へ lastinch 降下で置く物理」を検証・データ生成する。
（VLM 二値判定 + retry 被せは後段ジョブ。ここは orchestrator が呼ぶ 1 スキル呼び出し
 = pick_place(topping, goal) の実行体に相当する。）

方式（lastinch_spike_job + rollout_job の合成）:
  1. RL ポリシーで静定ホバー（rollout_job FINISH=script / lastinch と同一）
  2. 微分 IK で goal（実測ホバー点 + GOAL_OFFSET、丼内の到達可能ディスク r≦1.8cm）へ小移動
  3. 降下: トッピング底面をスープ面 z=0.012 すれすれへ（落下距離≒0 で転がり抑制）
  4. トッピング別開放:
     - RELEASE_TILT>0: wrist_flex 適応傾け（両ジョー接触が切れるまで）= ナルト等の薄物
     - RELEASE_FRIC>0: 開放直前に低摩擦へ（豚脂の見立て）= チャーシュー（可動ジョー載り対策）
     - 両者 0: 素の開放 = 味玉/メンマ/球
  5. 退避 → check_success（丼ゾーン内）→ 俯瞰 + サイド動画

丼は GYOZA_BOWL="x,y"（既定=到達可能ディスク中心）で env に差し込む（bowl.patch_bowl）。

実行:
    hf jobs uv run --flavor t4-small --timeout 2h --secrets HF_TOKEN \
        -v hf://buckets/YUGOROU/gyoza-sim:/gyoza --env GYOZA_DATA=/gyoza \
        --env MODEL=outputs/rl_v3a-warm-perf/best/best_model.zip \
        --env RUN=bowl_naruto --env EPISODES=10 --env VIDEO_N=10 \
        --env GYOZA_TOMATO=naruto --env RELEASE_TILT=25 \
        jobs/bowl_demo_job.py

    # チャーシュー（豚脂リリース）:
    #   --env GYOZA_TOMATO=chashu --env RELEASE_FRIC=0.12 --env RELEASE_TILT=35

env vars: MODEL / RUN / EPISODES / VIDEO_N / SEED / GYOZA_TOMATO
         / GOAL_OFFSET="dx,dy"(既定 "0,0.02"=ディスク中心) / DISK_R(既定0.0)=goal を
           ディスク内でランダム化する半径（0=中心固定・検証用）
         / RELEASE_TILT / RELEASE_FRIC / GRIP_OPEN(既定90)
"""

import json
import os
import pathlib
import subprocess
import sys

DATA = pathlib.Path(os.environ.get("GYOZA_DATA", "/gyoza"))
CODE = DATA / "code"

SOUP_TOP = 0.030   # bowl.py の patch_bowl(soup_z) と一致させる（降下目標＝スープ面）


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
    import mujoco
    import numpy as np
    from stable_baselines3 import PPO

    from gyoza.envs.pick_place import JOINTS, VEG_HEIGHTS
    from gyoza.envs.rl_pick_place import RLPickPlaceEnv

    model_rel = os.environ["MODEL"]
    run = os.environ.get("RUN", "bowl_demo")
    episodes = int(os.environ.get("EPISODES", "10"))
    video_n = int(os.environ.get("VIDEO_N", "10"))
    seed = int(os.environ.get("SEED", "0"))
    topping = os.environ.get("GYOZA_TOMATO", "sphere")
    goal_off = np.array([float(v) for v in os.environ.get("GOAL_OFFSET", "0,0.02").split(",")])
    disk_r = float(os.environ.get("DISK_R", "0.0"))
    release_tilt = float(os.environ.get("RELEASE_TILT", "0.0"))
    release_fric = float(os.environ.get("RELEASE_FRIC", "0.0"))
    grip_open = float(os.environ.get("GRIP_OPEN", "90.0"))
    out = DATA / "outputs" / "bowl_demo" / run
    out.mkdir(parents=True, exist_ok=True)

    # 丼中心 = 到達可能ディスク中心（実測ホバー点 anchor + goal_off）。
    # anchor はエピソード毎に ±2mm 揺れるが body_pos は定数なので代表値 (0.204,-0.118) を使う。
    ANCHOR0 = np.array([0.204, -0.118])
    bowl_c = ANCHOR0 + goal_off
    os.environ["GYOZA_BOWL"] = f"{bowl_c[0]:.4f},{bowl_c[1]:.4f}"

    # トッピング底面がスープ面に載るときの body 中心 z（メッシュは body 中心が上下中心）
    half_h = VEG_HEIGHTS.get(topping, 0.032) / 2 if topping != "sphere" else 0.016
    rest_z = SOUP_TOP + half_h

    MOVE_STEP = 0.003
    DQ_MAX = 0.02
    SETTLE_N = 5
    MOVE_TIMEOUT = 120
    DROP_CLEAR = 0.004

    model = PPO.load(str(DATA / model_rel), device="cpu")
    env = RLPickPlaceEnv(seed=seed, render_obs=True)
    CTRL_LO, CTRL_HI = env._ctrl_lo.copy(), env._ctrl_hi.copy()
    # RL obs の皿座標は学習時の値に凍結（ホバー挙動を完全保存。lastinch/rollout と同一）
    FROZEN_GOAL = np.array([0.26, -0.07, 0.014])
    env.env.plate_pos = lambda: FROZEN_GOAL.copy()
    m, d = env.env.model, env.env.data
    arm_dofs = [m.joint(f"follower_{j}").dofadr[0] for j in JOINTS[:5]]
    tcp_site = env.env._tcp_site
    rng = np.random.default_rng(seed + 7)
    jacp = np.zeros((3, m.nv))
    print(f"[bowl_demo] topping={topping} bowl@{bowl_c} rest_z={rest_z:.4f} "
          f"tilt={release_tilt} fric={release_fric} out={out}", flush=True)

    def in_hand() -> bool:
        # 接触フラグは押し下げ・移動で瞬断するので実体ベース（ジョー中点からの距離）
        return float(np.linalg.norm(env.env.tomato_pos() - env.env.jaw_mid_pos())) < 0.06

    def ik_step(tgt, dx):
        mujoco.mj_jacSite(m, d, jacp, None, tcp_site)
        dq = np.clip(np.linalg.pinv(jacp[:, arm_dofs], rcond=1e-4) @ dx, -DQ_MAX, DQ_MAX)
        tgt[:5] = np.clip(tgt[:5] + dq, CTRL_LO[:5], CTRL_HI[:5])
        env.env.step(np.rad2deg(tgt))

    results = []
    for ep in range(episodes):
        obs, _ = env.reset()
        frames_o, frames_s = [], []
        rec = dict(episode=ep)

        def snap():
            if ep < video_n:
                frames_o.append(env.env.render("overhead").copy())
                frames_s.append(env.env.render("side").copy())

        # --- phase 1: RL で静定ホバー ---
        hover, prev_tom = 0, env.env.tomato_pos()
        for t in range(250):
            a, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(a)
            snap()
            tom = env.env.tomato_pos()
            settled = float(np.linalg.norm(tom - prev_tom)) < 0.004
            prev_tom = tom
            hover = hover + 1 if (info["grasped"] and info["d_place"] < 0.12 and settled) else 0
            if trunc or hover >= 20:
                break
        if hover < 20:
            rec.update(stage="rl_fail", is_success=False)
            results.append(rec)
            print(f"[ep {ep:03d}] {rec}", flush=True)
            continue

        # --- phase 2: 微分 IK で goal へ小移動 ---
        anchor = env.env.tomato_pos()[:2].copy()
        theta, rad = rng.uniform(0, 2 * np.pi), disk_r * np.sqrt(rng.uniform(0, 1))
        goal = anchor + goal_off + rad * np.array([np.cos(theta), np.sin(theta)])
        z_hold = env.env.tcp_pos()[2]
        tgt = env._target.copy()
        ok, dropped, settle = False, False, 0
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
            ik_step(tgt, dx)
            snap()
        rec.update(stage="move", move_converged=bool(ok), dropped_in_move=bool(dropped))

        # --- phase 2.5: 降下（底面をスープ面すれすれへ）---
        if not dropped:
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
                    break  # 停滞（姿勢再構成の谷）: 押し続けず現高で開放
                dx = np.zeros(3)
                dx[:2] = np.clip(goal - tom[:2], -MOVE_STEP, MOVE_STEP)
                dx[2] = np.clip(e_z, -MOVE_STEP, MOVE_STEP)
                ik_step(tgt, dx)
                snap()
            rec.update(descend_z=round(float(env.env.tomato_pos()[2]), 4))

        # --- phase 3: トッピング別開放 ---
        if not dropped:
            if release_fric > 0:            # 豚脂: 開放直前に把持対象を低摩擦へ
                for gid in env.env._tomato_geom_ids:
                    env.env.model.geom_friction[gid, 0] = release_fric
            for k in range(1, 16):          # グリッパー漸進開放 0.5s
                tgt[5] = np.deg2rad(-10.0 + (grip_open + 10.0) * (k / 15))
                env.env.step(np.rad2deg(tgt))
                snap()
            tgt[5] = np.deg2rad(grip_open)
            if release_tilt > 0:            # 薄物/載り対策: 両ジョー接触が切れるまで適応傾け
                w0, tilt = tgt[3], 0.0
                while tilt < np.deg2rad(release_tilt):
                    tilt += np.deg2rad(1.25)
                    tgt[3] = w0 + tilt
                    env.env.step(np.rad2deg(tgt))
                    snap()
                for _ in range(40):
                    if not any(env.env.jaw_contacts()):
                        break
                    tilt = min(tilt + np.deg2rad(1.25), np.deg2rad(release_tilt + 50.0))
                    tgt[3] = w0 + tilt
                    env.env.step(np.rad2deg(tgt))
                    snap()
                for _ in range(15):
                    env.env.step(np.rad2deg(tgt))
                    snap()
            else:                           # 素の開放: 真上へ 3cm 抜いてから
                z_up = env.env.tcp_pos()[2] + 0.03
                for _ in range(25):
                    dx = np.array([0.0, 0.0, np.clip(z_up - env.env.tcp_pos()[2], -MOVE_STEP, MOVE_STEP)])
                    ik_step(tgt, dx)
                    snap()
            # 退避（肩引き）→ 静定
            tgt[1] -= np.deg2rad(25.0)
            for _ in range(45):
                env.env.step(np.rad2deg(tgt))
                snap()
                if env.env.check_success():
                    break

        final = env.env.tomato_pos()
        plate = env.env.data.site_xpos[env.env._plate_site]
        rec.update(is_success=bool(env.env.check_success()),
                   d_bowl=round(float(np.linalg.norm(final[:2] - plate[:2])), 4),
                   final_z=round(float(final[2]), 4), dropped=bool(dropped))
        results.append(rec)
        print(f"[ep {ep:03d}] {rec}", flush=True)
        if frames_o:
            imageio.mimsave(out / f"ep{ep:03d}_overhead.mp4", frames_o, fps=30)
            imageio.mimsave(out / f"ep{ep:03d}_side.mp4", frames_s, fps=30)

    placed = [r for r in results if r.get("stage") == "move"]
    summary = dict(
        topping=topping, bowl_center=[round(float(v), 4) for v in bowl_c],
        episodes=episodes,
        success_rate=round(sum(r["is_success"] for r in results) / episodes, 3),
        hover_rate=round(len(placed) / episodes, 3),
        mean_d_bowl=round(float(np.mean([r["d_bowl"] for r in results if "d_bowl" in r])), 4)
        if any("d_bowl" in r for r in results) else None,
        per_episode=results,
    )
    (out / "stats.json").write_text(json.dumps(summary, indent=2))
    print("[bowl_demo] DONE", json.dumps({k: v for k, v in summary.items() if k != "per_episode"}), flush=True)


if __name__ == "__main__":
    main()
