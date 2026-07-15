# /// script
# requires-python = ">=3.11"
# dependencies = ["mujoco", "numpy", "lerobot==0.4.4", "torch", "torchvision", "imageio[ffmpeg]", "huggingface_hub"]
# ///
"""選別つきクープ盛り付け — retry ablation 測定ジョブ（docs/coupe-protocol.md 準拠）。

2条件:
  A（オーケストレーションなし）: 各配置1回試行・検証なし
  B（VLM判定+retry≤2）: 配置後にクープクロップ before/after を Kimi 二値判定、no なら retry

実行体 = 果物別特化 ACT（ジオラマ方式: ライブ1個+静物）。駆動はスクリプト・決定論
（ablation の対象は検証・再試行レイヤ。エージェント LLM は測定に使わない）。

実行:
    hf jobs uv run --flavor t4-small --timeout 4h --secrets HF_TOKEN \
        --label name=coupe-ablation-s42 --label project=gyoza \
        -v hf://buckets/YUGOROU/gyoza-sim:/gyoza --env GYOZA_DATA=/gyoza \
        --env RUN=coupe_ablation_s42 --env SERIES=25 --env SEED=42 \
        --env CONDITIONS=A,B jobs/coupe_ablation_job.py

env vars: RUN / SERIES(系列数) / SEED / CONDITIONS("A,B") / VIDEO_N(既定5) / MAX_STEPS(既定400)
         / POLICY_SHIRATAMA / POLICY_GRAPE / POLICY_CHERRY(既定 YUGOROU/act_gyoza_{shiratama,grape,cherry3})
         / VLM_MODEL(既定 moonshotai/Kimi-K2.6)
         / JUDGE_MODE(yesno=v1凍結 | count=v2凍結 | count_split=v3候補: 単一画像カウント×2
           コール+色弁別ヒント。判定失敗・API全滅は no 扱いを継承)
         / MASK_STATICS(既定0。1 で観測マスキング: 静物を geom group 5 に置き、policy 入力
           からは不可視・VLM 判定/動画用の第2レンダラには可視。pick_place(object) の対象指定
           を観測マスクで実装 — 実機のセグメンテーションマスク相当。2026-07-10 診断で同種
           静物・密集シーンが把持を崩壊させると確定したことへの対処)
"""

import base64
import io
import json
import os
import pathlib
import subprocess
import sys

DATA = pathlib.Path(os.environ.get("GYOZA_DATA", "/gyoza"))
CODE = DATA / "code"

TASK = "pick the tomato and place it on the plate"
SEQUENCE = ["shiratama", "shiratama", "grape", "grape", "cherry_stage3_red"]
RECIPE = {"shiratama": 2, "grape": 2, "cherry_stage3_red": 1}
NEVER_PICK = ["cherry_stage1_green", "cherry_stage2_yellowpink"]
COUPE_CENTER = (0.204, -0.118)
COUPE_FLOOR_Z = 0.058
BOARD_Z = 0.015
LIVE_X, LIVE_Y = (-0.06, 0.06), (-0.16, -0.00)   # 訓練時のライブ球ランダム化領域


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


def png_b64(rgb) -> str:
    import imageio
    buf = io.BytesIO()
    imageio.imwrite(buf, rgb, format="png")
    return base64.b64encode(buf.getvalue()).decode()


def main():
    subprocess.run("apt-get update -qq && apt-get install -y -qq libegl1 libgles2 libosmesa6 ffmpeg",
                   shell=True, capture_output=True)
    os.environ["MUJOCO_GL"] = pick_gl_backend()
    os.environ["GYOZA_COUPE"] = "1"   # クープシーン（中心は patch_coupe 既定 = 皿位置）
    sys.path.insert(0, str(CODE))

    import imageio
    import mujoco
    import numpy as np
    import torch

    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors
    from huggingface_hub import InferenceClient
    from gyoza.envs.pick_place import PickPlaceEnv, VEG_HEIGHTS

    run = os.environ.get("RUN", "coupe_ablation")
    n_series = int(os.environ.get("SERIES", "25"))
    # 系列シャード（seed*10000+si で系列毎に決定論なので分割しても同一系列を再現）
    s_start = int(os.environ.get("SERIES_START", "0"))
    s_end = int(os.environ.get("SERIES_END", str(n_series)))  # 排他的上限
    seed = int(os.environ.get("SEED", "42"))
    conditions = os.environ.get("CONDITIONS", "A,B").split(",")
    video_n = int(os.environ.get("VIDEO_N", "5"))
    max_steps = int(os.environ.get("MAX_STEPS", "400"))
    vlm_model = os.environ.get("VLM_MODEL", "moonshotai/Kimi-K2.6")
    mask_statics = os.environ.get("MASK_STATICS", "0") == "1"
    out = DATA / "outputs" / "evals" / run
    out.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy_repos = {
        "shiratama": os.environ.get("POLICY_SHIRATAMA", "YUGOROU/act_gyoza_shiratama"),
        "grape": os.environ.get("POLICY_GRAPE", "YUGOROU/act_gyoza_grape"),
        "cherry_stage3_red": os.environ.get("POLICY_CHERRY", "YUGOROU/act_gyoza_cherry3"),
    }
    policies = {}
    for fruit, repo in policy_repos.items():
        pol = ACTPolicy.from_pretrained(repo).to(device).eval()
        cfg = PreTrainedConfig.from_pretrained(repo)
        pre, post = make_pre_post_processors(cfg, pretrained_path=repo)
        policies[fruit] = (pol, pre, post)
        print(f"[policy] {fruit} <- {repo}", flush=True)
    vlm = InferenceClient(api_key=os.environ.get("HF_TOKEN"), provider="auto")

    def sample_layout(rng):
        """7 果物の盤上レイアウト。把持対象5個はライブ領域内・pairwise≥4.5cm、
        未熟緑・中間黄桃はライブ領域外の距離物レンジ。"""
        placed = []
        def sample(xr, yr, dmin):
            for _ in range(500):
                p = (rng.uniform(*xr), rng.uniform(*yr))
                if all(np.hypot(p[0] - q[0], p[1] - q[1]) > dmin for q in placed):
                    return p
            raise RuntimeError("layout sampling failed")
        layout = {}
        for i, fruit in enumerate(SEQUENCE):
            p = sample(LIVE_X, LIVE_Y, 0.045)
            placed.append(p)
            layout[f"{fruit}#{i}"] = p
        for fruit in NEVER_PICK:
            for _ in range(500):
                p = (rng.uniform(-0.08, 0.16), rng.uniform(-0.17, 0.05))
                in_live = LIVE_X[0] - 0.02 < p[0] < LIVE_X[1] + 0.02 and \
                          LIVE_Y[0] - 0.02 < p[1] < LIVE_Y[1] + 0.02
                if not in_live and all(np.hypot(p[0] - q[0], p[1] - q[1]) > 0.04 for q in placed):
                    break
            placed.append(p)
            layout[f"{fruit}#x"] = p
        return layout

    def build_env(live_fruit, live_xy, board_statics, coupe_statics, rng):
        """ライブ1個+静物の env を構築し、ライブ果物を指定位置へテレポート・整定。"""
        statics = ([(nm, xy, yaw, BOARD_Z) for nm, xy, yaw in board_statics]
                   + [(nm, xy, yaw, COUPE_FLOOR_Z) for nm, xy, yaw in coupe_statics])
        os.environ["GYOZA_TOMATO"] = live_fruit
        env = PickPlaceEnv(seed=int(rng.integers(1 << 31)), render_obs=True, statics=statics,
                           statics_group=5 if mask_statics else 0)
        env.reset()
        tq = env._tomato_qpos
        z0 = BOARD_Z + VEG_HEIGHTS[live_fruit] / 2 + 0.005
        env.data.qpos[tq:tq + 7] = [live_xy[0], live_xy[1], z0, 1, 0, 0, 0]
        env.data.qvel[:] = 0
        mujoco.mj_forward(env.model, env.data)
        for _ in range(50):
            mujoco.mj_step(env.model, env.data)
        return env

    def coupe_crop(env, frame_hi, scale):
        from gyoza.envs.place_skill import project_bowl_bbox
        x0, y0, x1, y1 = project_bowl_bbox(env, COUPE_CENTER, r=0.075)
        s = scale
        return frame_hi[int(y0 * s):int(y1 * s), int(x0 * s):int(x1 * s)]

    judge_mode = os.environ.get("JUDGE_MODE", "yesno")  # yesno(v1凍結) | count(v2凍結) | count_split(v3候補)
    FRUIT_LABEL = {"shiratama": "white shiratama dumpling", "grape": "purple grape",
                   "cherry_stage3_red": "ripe red cherry"}
    # v3: 色弁別ヒント（grape 誤カウント FN 114/193 への対処）
    FRUIT_HINT = {
        "shiratama": "Shiratama are matte WHITE spheres. Do not count purple grapes or red cherries.",
        "grape": "Grapes are DARK PURPLE, almost black, small spheres. Do not count red cherries "
                 "(bright red) or white shiratama. Grapes may be hard to see against the glass.",
        "cherry_stage3_red": "Ripe cherries are BRIGHT RED. Do not count dark purple grapes "
                             "(darker, almost black) or white shiratama.",
    }

    def vlm_call(content):
        """API 障害（504/空応答）は判定でなくインフラ → 指数バックオフで吸収。
        全滅時は None を返し、呼び手はエラー文字列を数字パースしない（v2 では
        504 エラー文中の '504' '3' を before/after カウントと誤読していた）。"""
        import time
        for attempt in range(5):
            try:
                r = vlm.chat.completions.create(
                    model=vlm_model,
                    messages=[{"role": "user", "content": content}], max_tokens=8192)
                text = (r.choices[0].message.content or "").strip()
                if text:   # 空応答（Kimi thinking の枯渇）もリトライ対象
                    return text
            except Exception as e:
                print(f"    (api error att{attempt}: {str(e)[:80]})", flush=True)
            time.sleep(min(4 * 2 ** attempt, 60))
        return None

    def img_part(b64):
        return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}

    def count_single(img_b64, fruit):
        """単一画像の色別カウント（スモーク③で実証済みの得意形）。"""
        import re
        prompt = (f"Close-up top-down photo of a glass coupe. Count the {FRUIT_LABEL[fruit]}s "
                  f"that are INSIDE the coupe. {FRUIT_HINT[fruit]} "
                  f"End your answer with a single line: COUNT=<integer>")
        text = vlm_call([{"type": "text", "text": prompt}, img_part(img_b64)])
        if text is None:
            return None, "(api exhausted)"
        m = re.search(r"COUNT\s*=\s*(\d+)", text)
        if m:
            return int(m.group(1)), text
        nums = re.findall(r"\d+", text)
        return (int(nums[-1]), text) if nums else (None, text)

    def judge(before_b64, after_b64, fruit):
        import re
        label = FRUIT_LABEL[fruit]
        if judge_mode == "count_split":
            # v3 候補: before/after を独立にカウントし差分はコード側で取る
            # （2枚比較の同数誤読 106/193 への対処。判定失敗は no 扱い=プロトコル継承）
            nb, tb = count_single(before_b64, fruit)
            na, ta = count_single(after_b64, fruit)
            if nb is None or na is None:
                return False, f"(parse fail) b:{str(tb)[:50]} / a:{str(ta)[:50]}"
            return na > nb, f"b={nb} a={na}"
        if judge_mode == "count":
            prompt = (f"Two close-up top-down photos of the SAME glass coupe. FIRST=before, "
                      f"SECOND=after. Count the number of {label}s that are INSIDE the coupe "
                      f"in each photo. Answer with exactly two integers separated by a space: "
                      f"<before_count> <after_count>")
        else:
            prompt = (f"Two close-up top-down photos of the SAME glass coupe. FIRST=before, "
                      f"SECOND=after an attempt to add a {label}. Did a NEW {label} successfully "
                      f"land inside the coupe between the two photos? Answer only yes or no.")
        text = vlm_call([{"type": "text", "text": prompt},
                         img_part(before_b64), img_part(after_b64)])
        if text is None:
            return False, "(api exhausted)"
        if judge_mode == "count":
            nums = re.findall(r"\d+", text)
            ok = len(nums) >= 2 and int(nums[-1]) > int(nums[-2])
            return ok, text
        return text.lower().lstrip().startswith("y"), text

    def rollout(env, fruit, frames_sink):
        pol, pre, post = policies[fruit]
        pol.reset()
        succ = False
        for t in range(max_steps):
            img = env.render("overhead")
            if frames_sink is not None:
                frames_sink.append(img.copy())
            batch = {
                "observation.images.overhead":
                    torch.from_numpy(img).permute(2, 0, 1).float().div(255).unsqueeze(0).to(device),
                "observation.state":
                    torch.from_numpy(env.state_deg().astype(np.float32)).unsqueeze(0).to(device),
                "task": [TASK],
            }
            batch = pre(batch)
            with torch.inference_mode():
                action = pol.select_action(batch)
            action = post(action)
            env.step(action.squeeze(0).cpu().numpy().astype(np.float64))
            if env.check_success():
                succ = True
                break
        return succ

    results = []
    for si in range(s_start, s_end):
        rng_layout = np.random.default_rng(seed * 10000 + si)
        layout = sample_layout(rng_layout)
        for cond in conditions:
            rng = np.random.default_rng(seed * 10000 + si)   # 条件間で同一シード列
            board = {k: v for k, v in layout.items()}         # 残っている盤上果物
            coupe = []                                        # commit 済み (fruit, xy, yaw)
            frames = [] if si < video_n else None
            steps_log = []
            for k, fruit in enumerate(SEQUENCE):
                key = f"{fruit}#{k}"
                max_att = 1 if cond == "A" else 3
                placed_ok = False
                for att in range(max_att):
                    live_xy = board[key] if att == 0 else (
                        rng.uniform(*LIVE_X), rng.uniform(*LIVE_Y))  # retry は位置再サンプル
                    board_statics = [(kk.split("#")[0], xy, 0.0)
                                     for kk, xy in board.items() if kk != key]
                    env = build_env(fruit, live_xy, board_statics, coupe, rng)
                    hi_r = mujoco.Renderer(env.model, 720, 960)
                    # フルシーン用オプション: マスキング時も静物(group 5)を判定・動画に映す
                    # （group 4 は SO-101 の衝突 geom が使用済みのため不可）
                    full_opt = mujoco.MjvOption()
                    full_opt.geomgroup[5] = 1
                    def render_hi():
                        hi_r.update_scene(env.data, camera="overhead", scene_option=full_opt)
                        return hi_r.render()
                    before = coupe_crop(env, render_hi(), 1.5)
                    gt = rollout(env, fruit, frames)
                    after = coupe_crop(env, render_hi(), 1.5)
                    landing = env.tomato_pos()
                    verdict, raw = (None, "")
                    if cond == "B":
                        verdict, raw = judge(png_b64(before), png_b64(after), fruit)
                    steps_log.append(dict(series=si, cond=cond, step=k, fruit=fruit,
                                          attempt=att, gt=bool(gt),
                                          vlm=verdict, vlm_raw=raw[:120],
                                          landing=[round(float(v), 4) for v in landing[:2]]))
                    print(f"[s{si:02d} {cond} step{k} att{att}] {fruit} gt={gt} vlm={verdict}",
                          flush=True)
                    hi_r.close()
                    env.close()
                    # 物理の持ち越し（commit）= GT。判断ではなく物理継続性のエミュレーション
                    # （両条件同一）。判断層（retry するか）には GT を一切渡さない
                    if gt:
                        coupe.append((fruit, (float(landing[0]), float(landing[1])),
                                      float(rng.uniform(0, 2 * np.pi))))
                    # retry 判断: A は常に1回で終了、B は VLM 判定 yes で終了
                    # （VLM 偽陰性なら GT 成功済みでも retry → 過剰投入がタスク失敗に現れる）
                    if cond == "A" or verdict:
                        placed_ok = True
                        break
                del board[key]   # この工程の果物は消費（成否・条件によらず次工程へ）
            counts = {}
            for f, _, _ in coupe:
                counts[f] = counts.get(f, 0) + 1
            task_ok = counts == RECIPE
            results.append(dict(series=si, cond=cond, task_success=bool(task_ok),
                                placed=counts, steps=steps_log))
            print(f"[series {si:02d} {cond}] task_success={task_ok} placed={counts}", flush=True)
            if frames:
                imageio.mimsave(out / f"s{si:02d}_{cond}.mp4", frames, fps=30)

    summary = {}
    for cond in conditions:
        rs = [r for r in results if r["cond"] == cond]
        atts = [s for r in rs for s in r["steps"]]
        gt_first = [s for s in atts if s["attempt"] == 0]
        summary[cond] = dict(
            task_success_rate=sum(r["task_success"] for r in rs) / len(rs),
            per_attempt_gt=sum(s["gt"] for s in atts) / len(atts),
            first_attempt_gt=sum(s["gt"] for s in gt_first) / len(gt_first),
            n_attempts=len(atts),
            vlm_gt_agreement=(sum((s["vlm"] == s["gt"]) for s in atts if s["vlm"] is not None)
                              / max(1, sum(1 for s in atts if s["vlm"] is not None))),
        )
    # bucket FUSE は空ディレクトリを実体化しない — 動画を書かないシャード（VIDEO_N 超）では
    # 冒頭の mkdir が消えるため、書き込み直前に再作成する
    out.mkdir(parents=True, exist_ok=True)
    (out / "stats.json").write_text(json.dumps(dict(seed=seed, series=n_series,
                                                    summary=summary, results=results), indent=2))
    print("[ablation] DONE", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
