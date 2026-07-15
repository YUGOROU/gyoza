"""pick_place スキルの実行体（RL 静定ホバー → lastinch 降下 → トッピング別開放）。

盛り付けデモの orchestrator（jobs/bowl_assemble_job.py）が 1 スキル呼び出し
= place_topping(topping, goal) として使う。bowl_demo_job.py の単一トッピング検証で
確立した手順を関数化したもの（RL はハイブリッド expert = 決定論ホバー、最後の数 cm を
微分 IK で goal へ、平底トッピングは適応傾けで固定/可動ジョーから滑落）。

俯瞰カメラは固定（c2w≒identity）なので丼領域の pixel bbox は投影計算でクロップできる
（VLM 二値判定は丼クロップに対して行う = 位置弁別が効く）。
"""
from __future__ import annotations

import numpy as np
import mujoco

from gyoza.envs.pick_place import JOINTS, VEG_HEIGHTS

MOVE_STEP = 0.003
DQ_MAX = 0.02
SETTLE_N = 5
MOVE_TIMEOUT = 120
DROP_CLEAR = 0.004
FROZEN_GOAL = np.array([0.26, -0.07, 0.014])   # RL obs の皿座標を学習時値に凍結


def project_bowl_bbox(env, center_xy, r: float = 0.075, cam: str = "overhead"):
    """固定俯瞰カメラで丼（中心 center_xy・半径 r）の pixel bbox を投影計算する。"""
    m, d = env.model, env.data
    cid = m.camera(cam).id
    c, R = d.cam_xpos[cid], d.cam_xmat[cid].reshape(3, 3)
    H, W = env.renderer.height, env.renderer.width
    f = (H / 2) / np.tan(np.deg2rad(m.cam_fovy[cid]) / 2)

    def proj(p):
        v = R.T @ (np.asarray(p) - c)
        return W / 2 + f * (v[0] / -v[2]), H / 2 - f * (v[1] / -v[2])

    pts = [proj((center_xy[0] + r * np.cos(a), center_xy[1] + r * np.sin(a), 0.03))
           for a in np.linspace(0, 2 * np.pi, 16)]
    us, vs = [p[0] for p in pts], [p[1] for p in pts]
    return (int(max(0, min(us))), int(max(0, min(vs))),
            int(min(W, max(us))), int(min(H, max(vs))))


def place_topping(env, model, topping: str, goal_xy, release_tilt: float,
                  soup_top: float, rng=None, grip_open: float = 90.0, capture=None) -> dict:
    """1 スキル実行。env は render_obs=True の RLPickPlaceEnv（呼び出し前に reset 済み）。

    goal_xy: 卓面の絶対座標（エージェントが「どこに置くか」で渡す）。到達可能域は RL ホバー
    到達点周辺 半径~1.8cm（呼び出し側でクランプ推奨）。
    返り値: dict(hover_ok, dropped, landing_xy, landing_z)。capture(frame) が与えられれば
    各制御ステップで俯瞰フレームを記録（連続デモ動画・retry の様子も含む）。
    """
    e = env.env
    e.plate_pos = lambda: FROZEN_GOAL.copy()
    m, d = e.model, e.data
    arm_dofs = [m.joint(f"follower_{j}").dofadr[0] for j in JOINTS[:5]]
    tcp = e._tcp_site
    jacp = np.zeros((3, m.nv))
    lo, hi = env._ctrl_lo, env._ctrl_hi
    half_h = VEG_HEIGHTS.get(topping, 0.032) / 2 if topping != "sphere" else 0.016
    rest_z = soup_top + half_h

    def snap():
        if capture is not None:
            capture(e.render("overhead"))

    def in_hand():
        return float(np.linalg.norm(e.tomato_pos() - e.jaw_mid_pos())) < 0.06

    def ik(tgt, dx):
        mujoco.mj_jacSite(m, d, jacp, None, tcp)
        dq = np.clip(np.linalg.pinv(jacp[:, arm_dofs], rcond=1e-4) @ dx, -DQ_MAX, DQ_MAX)
        tgt[:5] = np.clip(tgt[:5] + dq, lo[:5], hi[:5])
        e.step(np.rad2deg(tgt))

    # --- phase 1: RL 静定ホバー ---
    obs = env._obs()
    hover, prev = 0, e.tomato_pos()
    for _ in range(250):
        a, _ = model.predict(obs, deterministic=True)
        obs, _, _, trunc, info = env.step(a)
        snap()
        tom = e.tomato_pos()
        settled = float(np.linalg.norm(tom - prev)) < 0.004
        prev = tom
        hover = hover + 1 if (info["grasped"] and info["d_place"] < 0.12 and settled) else 0
        if trunc or hover >= 20:
            break
    if hover < 20:
        return dict(hover_ok=False, dropped=False,
                    landing_xy=e.tomato_pos()[:2].tolist(), landing_z=float(e.tomato_pos()[2]))

    # --- phase 2: 微分 IK で goal（絶対）へ ---
    goal = np.asarray(goal_xy, dtype=float)
    z_hold = e.tcp_pos()[2]
    tgt = env._target.copy()
    dropped, settle = False, 0
    for _ in range(MOVE_TIMEOUT):
        tom = e.tomato_pos()
        exy = goal - tom[:2]
        if not in_hand():
            dropped = True
            break
        if np.linalg.norm(exy) < 0.005:
            settle += 1
            if settle >= SETTLE_N:
                break
        else:
            settle = 0
        dx = np.zeros(3)
        n = np.linalg.norm(exy)
        dx[:2] = exy if n < MOVE_STEP else exy / n * MOVE_STEP
        dx[2] = np.clip(z_hold - e.tcp_pos()[2], -MOVE_STEP, MOVE_STEP)
        ik(tgt, dx)
        snap()

    # --- phase 2.5: 降下（底面をスープ面すれすれへ）---
    if not dropped:
        zh = []
        for _ in range(200):
            tom = e.tomato_pos()
            ez = (rest_z + DROP_CLEAR) - tom[2]
            if not in_hand():
                dropped = True
                break
            if abs(ez) < 0.002:
                break
            zh.append(tom[2])
            if len(zh) > 25 and zh[-26] - tom[2] < 0.001:
                break
            dx = np.zeros(3)
            dx[:2] = np.clip(goal - tom[:2], -MOVE_STEP, MOVE_STEP)
            dx[2] = np.clip(ez, -MOVE_STEP, MOVE_STEP)
            ik(tgt, dx)
            snap()

    # --- phase 3: トッピング別開放（適応傾け）→ 退避 ---
    if not dropped:
        for k in range(1, 16):
            tgt[5] = np.deg2rad(-10.0 + (grip_open + 10.0) * (k / 15))
            e.step(np.rad2deg(tgt))
            snap()
        tgt[5] = np.deg2rad(grip_open)
        if release_tilt > 0:
            w0, tilt = tgt[3], 0.0
            while tilt < np.deg2rad(release_tilt):
                tilt += np.deg2rad(1.25)
                tgt[3] = w0 + tilt
                e.step(np.rad2deg(tgt))
                snap()
            for _ in range(40):
                if not any(e.jaw_contacts()):
                    break
                tilt = min(tilt + np.deg2rad(1.25), np.deg2rad(release_tilt + 50.0))
                tgt[3] = w0 + tilt
                e.step(np.rad2deg(tgt))
                snap()
            for _ in range(15):
                e.step(np.rad2deg(tgt))
                snap()
        else:
            z_up = e.tcp_pos()[2] + 0.03
            for _ in range(25):
                dx = np.array([0.0, 0.0, np.clip(z_up - e.tcp_pos()[2], -MOVE_STEP, MOVE_STEP)])
                ik(tgt, dx)
                snap()
        tgt[1] -= np.deg2rad(25.0)
        for _ in range(45):
            e.step(np.rad2deg(tgt))
            snap()

    tom = e.tomato_pos()
    return dict(hover_ok=True, dropped=bool(dropped),
                landing_xy=tom[:2].tolist(), landing_z=float(tom[2]))
