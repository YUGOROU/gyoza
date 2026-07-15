# /// script
# requires-python = ">=3.11"
# dependencies = ["mujoco==3.10.0", "numpy", "gymnasium", "stable-baselines3", "torch", "imageio[ffmpeg]", "huggingface_hub"]
# ///
# 注: mujoco はローカル(.venv)と同一の 3.10.0 に固定。未ピンだとクラウドが別版を引き、
#     RL のカオス的閉ループ把持が版差で乱れる（2026-07-08 実測: 未ピンで chashu 把持
#     クラウド 20% vs ローカル/3.10.0 で 70%）。lerobot 0.4.4 固定と同じ理由。
"""RL エキスパートのロールアウト実行ジョブ — 挙動診断 兼 蒸留データ工場。

PPO zip を bucket から読み、決定論ポリシーで N エピソード実行。
- 全エピソード: 軌跡 npz（状態・絶対関節目標[deg]・トマト/皿座標・成功フラグ・診断情報）
- 先頭 VIDEO_N エピソード: 俯瞰カメラ mp4（30fps）
- stats.json: 成功率・把持率などの集計

描画があるため GPU flavor（t4-small）推奨（EGL。CPU flavor なら osmesa に自動フォールバック）。

実行:
    hf jobs uv run --flavor t4-small --timeout 2h --secrets HF_TOKEN \
        -v hf://buckets/YUGOROU/gyoza-sim:/gyoza --env GYOZA_DATA=/gyoza \
        --env MODEL=outputs/rl_v3a-warm-perf/best/best_model.zip \
        --env RUN=diag_v3a --env EPISODES=20 --env VIDEO_N=3 \
        jobs/rollout_job.py

env vars: MODEL(bucket 相対パス) / RUN / EPISODES / VIDEO_N / SEED / REWARD_VARIANT
         / FINISH=script — ハイブリッド expert モード。RL ポリシーで皿縁ホバー
           （grasped & d_place<0.12）まで到達後、焼き込み済み到達姿勢 REACH_Q へ
           関節補間 → グリッパー開放 → 退避 をスクリプトで完遂する。
           （v5 までの RL は皿縁 7.4cm ホバー止まり。FK 検証で到達可能と確認済みの
             姿勢に最後の 2cm だけ演出でなく決定論制御で運ぶ = 蒸留データ工場用）
"""

import json
import os
import pathlib
import subprocess
import sys

DATA = pathlib.Path(os.environ.get("GYOZA_DATA", "/gyoza"))
CODE = DATA / "code"


def pick_gl_backend() -> str:
    """サブプロセスで egl → osmesa の順に描画可否を試す（プロセス単位でしか切替不可）。"""
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
    raise RuntimeError("描画バックエンドなし（libegl1/libosmesa6 の apt install を確認）")


def main():
    subprocess.run("apt-get update -qq && apt-get install -y -qq libegl1 libgles2 libosmesa6 ffmpeg",
                   shell=True, capture_output=True)
    os.environ["MUJOCO_GL"] = pick_gl_backend()
    sys.path.insert(0, str(CODE))

    import imageio
    import numpy as np
    from stable_baselines3 import PPO

    from gyoza.envs.rl_pick_place import RLPickPlaceEnv

    model_rel = os.environ["MODEL"]
    run = os.environ.get("RUN", "diag")
    episodes = int(os.environ.get("EPISODES", "20"))
    video_n = int(os.environ.get("VIDEO_N", "3"))
    seed = int(os.environ.get("SEED", "0"))
    scripted = os.environ.get("FINISH") == "script"
    out = DATA / "outputs" / "rollouts" / run
    out.mkdir(parents=True, exist_ok=True)

    # スクリプト仕上げ = ゴールバイアス補償 + 開放のみ:
    # 開ループ FK 姿勢 3 案・閉ループサーボは全て失敗（手首回転/初期ゲイン符号で弾く）。
    # RL の系統誤差（皿手前 7.4cm±2mm）を逆手に取り、obs の皿座標をずらして
    # ポリシー自身に皿中心上空へ運ばせる。腕のスクリプト制御は行わない。
    GRIP_OPEN = float(os.environ.get("GRIP_OPEN", "40.0"))
    # 薄物リリース補助（2026-07-08）: 平底ワーク（ナルト/チャーシュー）は開放しても
    # 固定ジョー上に乗ったまま退避で持ち上がる（smoke_*_v2 で機序確定: fixed jaw 接触が
    # 開放後も残存、肩引きで z 123→191mm にすくい上げ）。wrist_flex を +方向に傾けると
    # ジョー先端が下がり（FK 実測: +18° で TCP -50mm、xy ずれ <6mm）ワークが皿へ滑落する。
    # 0 なら従来挙動（トマト球の datagen 条件を変えないため opt-in）。
    RELEASE_TILT = float(os.environ.get("RELEASE_TILT", "0.0"))
    # 豚脂リリース（2026-07-08）: チャーシューは楕円形状で「可動ジョーの平らな上面に載る」
    # ため、傾けても可動ジョーごと外へ振れ z~90mm で貼り付き落ちない（v5=0% の真因）。
    # 運搬中は高摩擦(0.6)が把持に要るが、開放直前に低摩擦へ落とすと可動ジョーから滑落する。
    # 0 なら摩擦切替なし（従来挙動を保存）。チャーシューは 0.12 推奨（着地一貫 std 5-7mm）。
    RELEASE_FRIC = float(os.environ.get("RELEASE_FRIC", "0.0"))

    model = PPO.load(str(DATA / model_rel), device="cpu")
    env = RLPickPlaceEnv(seed=seed, render_obs=True)
    print(f"[rollout] model={model_rel} episodes={episodes} scripted={scripted} out={out}", flush=True)

    # 丼移設（"x,y"）: 豚脂リリースでチャーシューが滑落する着地点へ皿 body を移す。
    # 皿は contype=0（非衝突）・RL は皿 obs を凍結し無視するので把持・運搬は不変。
    # check_success は site（body に追従）で判定するため、着地点＝丼で成功評価できる。
    # body_pos はモデル定数で reset を跨いで保持される。
    bowl_env = os.environ.get("BOWL_XY")
    if bowl_env:
        bx, by = (float(v) for v in bowl_env.split(","))
        env.env.model.body_pos[env.env.model.body("plate").id, :2] = [bx, by]
        print(f"[rollout] bowl moved to ({bx},{by})", flush=True)

    # --- ゴール凍結（FINISH=script 時）---
    # 確定した失敗機序（2026-07-06）: RL ポリシーは皿座標 obs を実質無視し（皿固定
    # 学習の縮退）、決定論的に毎回ほぼ同一点 (0.204,-0.118)±2mm でホバーする。
    # 対応として XML の皿をこの実測到達点へ移設済み。obs の皿座標は学習時の値に
    # 凍結してポリシー挙動を完全に保存する（test5 で obs 変化が把持を乱すのを確認済）。
    if scripted:
        FROZEN_GOAL = np.array([0.26, -0.07, 0.014])  # 学習時の皿 site 位置
        env.env.plate_pos = lambda: FROZEN_GOAL.copy()

    stats = []
    for ep in range(episodes):
        obs, _ = env.reset()
        states, targets, frames = [], [], []
        succ_any = False
        grasp_steps = 0
        hover = 0
        final = {}

        def record(target_deg):
            states.append(env.env.state_deg().copy())
            targets.append(np.asarray(target_deg, dtype=np.float64).copy())
            if ep < video_n:
                frames.append(env.env.render("overhead"))

        # --- phase 1: RL ポリシー（scripted 時はリム衝突ホバー = 静定把持で打ち切り）---
        prev_tom = env.env.tomato_pos()
        for t in range(250 if scripted else 300):
            a, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(a)
            record(np.rad2deg(env._target))
            succ_any = succ_any or info["is_success"]
            grasp_steps += info["grasped"]
            final = info
            if scripted:
                tom = env.env.tomato_pos()
                settled = float(np.linalg.norm(tom - prev_tom)) < 0.004  # リム押し付けの微振動は許容
                prev_tom = tom
                hover = hover + 1 if (info["grasped"] and info["d_place"] < 0.12 and settled) else 0
            if trunc or (scripted and hover >= 20):
                break

        # --- phase 2: スクリプト仕上げ（補間 → 開放 → 退避）---
        # --- phase 2: 皿（実測到達点に移設済み）の上空でグリッパー開放 → 退避 ---
        if scripted and hover >= 20:
            real_plate = env.env.data.site_xpos[env.env._plate_site][:2].copy()
            if RELEASE_FRIC > 0:            # 豚脂リリース: 開放直前に把持対象を低摩擦へ
                for gid in env.env._tomato_geom_ids:
                    env.env.model.geom_friction[gid, 0] = RELEASE_FRIC
            tgt = np.rad2deg(env._target).astype(np.float64).copy()
            for k in range(1, 16):          # グリッパー開放 0.5s
                tgt[5] = -10.0 + (GRIP_OPEN + 10.0) * (k / 15)
                env.env.step(tgt)
                record(tgt)
            tgt[5] = GRIP_OPEN
            if RELEASE_TILT > 0:            # 薄物: ジョー先端を下に傾けて滑落させる
                # 固定角では不足（v3: ホバー姿勢の個体差で 25° でもジョー中腹に残留 →
                # 退避で再すくい上げ）。fixed jaw 接触が切れるまで適応的に傾ける。
                w0 = tgt[3]
                tilt = 0.0
                while tilt < RELEASE_TILT:  # まず指定角まで漸進
                    tilt += 1.25
                    tgt[3] = w0 + tilt
                    env.env.step(tgt)
                    record(tgt)
                for _ in range(40):         # 接触が残るなら +50° を上限に追加傾け
                    # 両ジョーを監視: ナルトは固定ジョー・チャーシューは可動ジョーに残る
                    # ため、どちらの接触も切れるまで傾ける（旧: 固定ジョー[0] のみ）。
                    if not any(env.env.jaw_contacts()):
                        break
                    tilt = min(tilt + 1.25, RELEASE_TILT + 50.0)
                    tgt[3] = w0 + tilt
                    env.env.step(tgt)
                    record(tgt)
                for _ in range(15):         # 滑落・静定待ち
                    env.env.step(tgt)
                    record(tgt)
            tgt[1] -= 25.0                  # 肩を引いて退避、トマトを静定させる
            for _ in range(45):
                env.env.step(tgt)
                record(tgt)
                if env.env.check_success():
                    break
            succ = env.env.check_success()
            final = dict(final, is_success=succ, d_place=float(
                np.linalg.norm(env.env.tomato_pos()[:2] - real_plate)))
            succ_any = succ_any or succ

        rec = dict(episode=ep, success_final=bool(final["is_success"]), success_any=bool(succ_any),
                   grasp_steps=int(grasp_steps), end_d_place=round(final["d_place"], 4),
                   hover_reached=bool(hover >= 20), steps=len(states))
        stats.append(rec)
        print(f"[ep {ep:03d}] {rec}", flush=True)

        np.savez_compressed(out / f"ep{ep:03d}.npz",
                            states_deg=np.array(states, dtype=np.float32),
                            targets_deg=np.array(targets, dtype=np.float32),
                            success_final=final["is_success"], success_any=succ_any)
        if frames:
            imageio.mimsave(out / f"ep{ep:03d}.mp4", frames, fps=30)

    summary = dict(
        model=model_rel, episodes=episodes,
        success_rate_final=sum(s["success_final"] for s in stats) / episodes,
        success_rate_any=sum(s["success_any"] for s in stats) / episodes,
        grasp_rate=sum(s["grasp_steps"] > 30 for s in stats) / episodes,
        mean_end_d_place=round(float(np.mean([s["end_d_place"] for s in stats])), 4),
        per_episode=stats,
    )
    (out / "stats.json").write_text(json.dumps(summary, indent=2))
    print("[rollout] DONE", json.dumps({k: v for k, v in summary.items() if k != "per_episode"}), flush=True)


if __name__ == "__main__":
    main()
