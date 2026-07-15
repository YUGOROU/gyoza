# /// script
# requires-python = ">=3.11"
# dependencies = ["mujoco", "numpy", "gymnasium", "stable-baselines3", "torch", "imageio[ffmpeg]", "huggingface_hub"]
# ///
"""ラストインチ制御スパイク — ゴール条件付き配置の技術検証。

問い: 静定ホバー（把持済み・皿上空）から、閉ループ微分 IK で ±4cm の小移動を行い
トマトを弾かずに指令点へ置けるか？（過去のスクリプト制御全滅は 7.4cm 運搬フェーズの
話。静定状態からの小摂動は条件が緩いはず、の検証）

方式:
  1. RL ポリシーで静定ホバーまで（rollout_job の FINISH=script と同一）
  2. goal = 実測ホバー到達点 + Uniform(disk, r≦OFFSET_R) をトライアル毎にサンプル
  3. 閉ループ: e = [goal_xy - tomato_xy, z_hold - tcp_z] → dq = pinv(J_tcp[:, arm5]) @ dx
     （1 ステップの dx は MOVE_STEP に制限、dq も DQ_MAX でクリップ）
  4. 収束（|e_xy| < 5mm が数ステップ）or タイムアウト → 開放 → 退避 → 静定
  5. 計測: 指令点からの誤差 / 把持維持 / 皿ゾーン内か

実行:
    hf jobs uv run --flavor t4-small --timeout 2h --secrets HF_TOKEN \
        -v hf://buckets/YUGOROU/gyoza-sim:/gyoza --env GYOZA_DATA=/gyoza \
        --env MODEL=outputs/rl_v3a-warm-perf/best/best_model.zip \
        --env RUN=lastinch_spike --env TRIALS=30 --env VIDEO_N=8 \
        jobs/lastinch_spike_job.py

env vars: MODEL / RUN / TRIALS / VIDEO_N / SEED / OFFSET_R(既定0.04)
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
    import mujoco
    import numpy as np
    from stable_baselines3 import PPO

    from gyoza.envs.pick_place import JOINTS
    from gyoza.envs.rl_pick_place import RLPickPlaceEnv

    model_rel = os.environ["MODEL"]
    run = os.environ.get("RUN", "lastinch_spike")
    trials = int(os.environ.get("TRIALS", "30"))
    video_n = int(os.environ.get("VIDEO_N", "8"))
    seed = int(os.environ.get("SEED", "0"))
    offset_r = float(os.environ.get("OFFSET_R", "0.04"))
    out = DATA / "outputs" / "spikes" / run
    out.mkdir(parents=True, exist_ok=True)

    MOVE_STEP = 0.003   # m / control step（xy 移動量上限）
    DQ_MAX = 0.02       # rad / control step（関節増分上限）
    SETTLE_N = 5        # |e_xy|<5mm をこのステップ数維持で収束
    MOVE_TIMEOUT = 120  # 制御ステップ（4 s）
    GRIP_OPEN = 40.0

    model = PPO.load(str(DATA / model_rel), device="cpu")
    env = RLPickPlaceEnv(seed=seed, render_obs=True)
    CTRL_LO, CTRL_HI = env._ctrl_lo.copy(), env._ctrl_hi.copy()
    FROZEN_GOAL = np.array([0.26, -0.07, 0.014])
    env.env.plate_pos = lambda: FROZEN_GOAL.copy()
    m, d = env.env.model, env.env.data
    arm_dofs = [m.joint(f"follower_{j}").dofadr[0] for j in JOINTS[:5]]
    tcp_site = env.env._tcp_site
    rng = np.random.default_rng(seed + 7)

    results = []
    for tr in range(trials):
        obs, _ = env.reset()
        frames = []
        rec = dict(trial=tr)

        def snap():
            if tr < video_n:
                frames.append(env.env.render("overhead").copy())

        # --- phase 1: RL で静定ホバーへ ---
        hover = 0
        prev_tom = env.env.tomato_pos()
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
            rec.update(stage="rl_fail")
            results.append(rec)
            print(f"[tr {tr:02d}] {rec}", flush=True)
            continue

        # --- phase 2: 微分 IK でゴールへ小移動 ---
        # v5 分析の確定知見: 降下は y>+0.015（奥側）で 6/6 完走・平均誤差 1.4cm、
        # -y（手前側）は姿勢再構成の谷で停滞（elbow 可動域）。ゴール円盤の中心を
        # 到達点 +y 2cm に置けば全ゴールが実行可能領域に入り、皿座標系では全方位カバー
        anchor = env.env.tomato_pos()[:2].copy()      # このエピソードの実測ホバー点
        disk_c = anchor + np.array([0.0, float(os.environ.get("DISK_Y", "0.02"))])
        theta = rng.uniform(0, 2 * np.pi)
        rad = offset_r * np.sqrt(rng.uniform(0, 1))
        goal = disk_c + rad * np.array([np.cos(theta), np.sin(theta)])
        z_hold = env.env.tcp_pos()[2]
        tgt = env._target.copy()                       # rad 6
        jacp = np.zeros((3, m.nv))

        def in_hand() -> bool:
            # 接触フラグは押し下げ・移動で1ステップ瞬断する（v2 の全滅はこれの誤判定）。
            # 実体ベース: トマトがジョー中点から離れたら真の落下。
            return float(np.linalg.norm(env.env.tomato_pos() - env.env.jaw_mid_pos())) < 0.05

        ok, dropped = False, False
        settle = 0
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
            J = jacp[:, arm_dofs]                      # 3x5
            dq = np.linalg.pinv(J, rcond=1e-4) @ dx
            dq = np.clip(dq, -DQ_MAX, DQ_MAX)
            # anti-windup: 停滞中に目標を積分し続けると実姿勢から乖離し開放・退避で暴れる
            tgt[:5] = np.clip(tgt[:5] + dq, CTRL_LO[:5], CTRL_HI[:5])
            env.env.step(np.rad2deg(tgt))
            snap()

        move_err = float(np.linalg.norm(env.env.tomato_pos()[:2] - goal))
        rec.update(stage="move", offset=[round(float(v), 4) for v in (goal - anchor)],
                   move_converged=bool(ok), dropped_in_move=bool(dropped),
                   move_err=round(move_err, 4))

        # --- phase 2.5: 降下（トマト底面を皿面すれすれへ。落下距離≒0 にして転がり抑制）---
        # スパイク v1 の結果: 把持中の位置決めは sub-mm だが、ホバー高度からの
        # 開放でバウンド・転がりが乗り settle_err 平均 4.7cm / within_2cm 0。
        if not dropped:
            DROP_CLEAR = 0.004                         # 開放時の底面クリアランス
            rest_z = 0.015 + 0.016                     # 皿面(≒マット+皿厚) + トマト半径 ≒ 静止中心 z
            # v3 実測: ホバー高度は最大 ~20cm → 60 steps × 3mm では降下が完走しない
            # （descend_z と settle_err に強い相関）。200 steps に拡張し終了理由を記録
            # 停滞検出: 姿勢再構成の谷で降下が止まったら押し続けず、その高さで開放する
            # （v4 で 200 steps 押し続けた群が大きく外した。押すほど姿勢が歪む）
            descend_done = False
            z_hist = []
            for t in range(200):
                tom = env.env.tomato_pos()
                e_z = (rest_z + DROP_CLEAR) - tom[2]
                if not in_hand():
                    dropped = True
                    break
                if abs(e_z) < 0.002:
                    descend_done = True
                    break
                z_hist.append(tom[2])
                if len(z_hist) > 25 and z_hist[-26] - tom[2] < 0.001:
                    break  # 25 steps で 1mm も下りていない = 停滞
                dx = np.zeros(3)
                dx[:2] = np.clip(goal - tom[:2], -MOVE_STEP, MOVE_STEP)  # xy は維持
                dx[2] = np.clip(e_z, -MOVE_STEP, MOVE_STEP)
                mujoco.mj_jacSite(m, d, jacp, None, tcp_site)
                dq = np.clip(np.linalg.pinv(jacp[:, arm_dofs], rcond=1e-4) @ dx, -DQ_MAX, DQ_MAX)
                tgt[:5] = np.clip(tgt[:5] + dq, CTRL_LO[:5], CTRL_HI[:5])
                env.env.step(np.rad2deg(tgt))
                snap()
            rec.update(descend_z=round(float(env.env.tomato_pos()[2]), 4),
                       descend_done=bool(descend_done), descend_steps=t + 1,
                       dropped_in_descend=bool(dropped and rec.get("dropped_in_move") is False))

        # --- phase 3: 開放 → 退避 → 静定計測 ---
        if not dropped:
            for k in range(1, 16):
                tgt[5] = np.deg2rad(-10.0 + (GRIP_OPEN + 10.0) * (k / 15))
                env.env.step(np.rad2deg(tgt))
                snap()
            # 真上へ 3cm 抜いてから退避（低位置からの肩引きはジョーがトマトを弾く）
            z_up = env.env.tcp_pos()[2] + 0.03
            for t in range(25):
                dx = np.array([0.0, 0.0, np.clip(z_up - env.env.tcp_pos()[2], -MOVE_STEP, MOVE_STEP)])
                mujoco.mj_jacSite(m, d, jacp, None, tcp_site)
                dq = np.clip(np.linalg.pinv(jacp[:, arm_dofs], rcond=1e-4) @ dx, -DQ_MAX, DQ_MAX)
                tgt[:5] = np.clip(tgt[:5] + dq, CTRL_LO[:5], CTRL_HI[:5])
                env.env.step(np.rad2deg(tgt))
                snap()
            tgt[1] -= np.deg2rad(25.0)
            for _ in range(45):
                env.env.step(np.rad2deg(tgt))
                snap()
            final = env.env.tomato_pos()
            err = float(np.linalg.norm(final[:2] - goal))
            rec.update(settle_err=round(err, 4),
                       within_2cm=bool(err < 0.02),
                       in_zone=bool(env.env.check_success()))
        results.append(rec)
        print(f"[tr {tr:02d}] {rec}", flush=True)
        if frames:
            imageio.mimsave(out / f"tr{tr:02d}.mp4", frames, fps=30)

    moved = [r for r in results if r.get("stage") == "move"]
    placed = [r for r in moved if "settle_err" in r]
    summary = dict(
        trials=trials, rl_hover_ok=len(moved),
        dropped_in_move=sum(r["dropped_in_move"] for r in moved),
        move_converged=sum(r["move_converged"] for r in moved),
        placed=len(placed),
        within_2cm=sum(r.get("within_2cm", False) for r in placed),
        in_zone=sum(r.get("in_zone", False) for r in placed),
        mean_settle_err=round(float(np.mean([r["settle_err"] for r in placed])), 4) if placed else None,
        per_trial=results,
    )
    (out / "stats.json").write_text(json.dumps(summary, indent=2))
    print("[spike] DONE", json.dumps({k: v for k, v in summary.items() if k != "per_trial"}), flush=True)


if __name__ == "__main__":
    main()
