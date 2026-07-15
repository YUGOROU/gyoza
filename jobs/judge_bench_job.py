# /// script
# requires-python = ">=3.11"
# dependencies = ["mujoco", "numpy", "imageio", "huggingface_hub"]
# ///
"""判定 VLM のオフラインベンチ用クロップペア生成（プロトコル v3 検証用）。

ablation v2 で VLM 偽陰性 49%（grape に集中）が判明。判定器の修正候補
（単一画像カウント×2 / プロンプト / クロップ拡大 / モデル交換）を測定前に
オフラインで検証・凍結するため、測定と同一光学系（720x960 第2レンダラ +
coupe_crop scale1.5 + group5 フルシーン）で GT ラベル付き before/after
ペアを生成する。

各ペア: クープ内容（白玉0-2・ぶどう0-2・さくらんぼ0-1）を設置 → before 描画
→ ライブ果物をクープ内へテレポート整定（added=True）or 盤上の別位置へ
（added=False）→ after 描画。判定対象果物は grape を過重サンプル。

実行:
    hf jobs uv run --flavor t4-small --timeout 1h --secrets HF_TOKEN \
        --label name=judge_bench_v1 --label project=gyoza \
        -v hf://buckets/YUGOROU/gyoza-sim:/gyoza --env GYOZA_DATA=/gyoza \
        --env RUN=judge_bench_v1 --env PAIRS=160 jobs/judge_bench_job.py

env vars: RUN / PAIRS(既定160) / SEED(既定7)
"""

import json
import os
import pathlib
import subprocess
import sys

DATA = pathlib.Path(os.environ.get("GYOZA_DATA", "/gyoza"))
CODE = DATA / "code"

COUPE_CENTER = (0.204, -0.118)
COUPE_FLOOR_Z = 0.058
BOARD_Z = 0.015
FRUITS = ["shiratama", "grape", "cherry_stage3_red"]
# grape 過重（FN 193中114 が grape）
QUERY_CYCLE = ["grape", "grape", "shiratama", "cherry_stage3_red"]


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
    subprocess.run("apt-get update -qq && apt-get install -y -qq libegl1 libgles2 libosmesa6",
                   shell=True, capture_output=True)
    os.environ["MUJOCO_GL"] = pick_gl_backend()
    os.environ["GYOZA_COUPE"] = "1"
    sys.path.insert(0, str(CODE))

    import imageio
    import mujoco
    import numpy as np

    from gyoza.envs.pick_place import PickPlaceEnv, VEG_HEIGHTS

    run = os.environ.get("RUN", "judge_bench_v1")
    n_pairs = int(os.environ.get("PAIRS", "160"))
    seed = int(os.environ.get("SEED", "7"))
    out = DATA / "outputs" / "evals" / run
    out.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)

    def sample_coupe_positions(n, existing=()):
        pts = list(existing)
        for _ in range(n):
            for _ in range(500):
                r = rng.uniform(0, 0.042)
                th = rng.uniform(0, 2 * np.pi)
                p = (COUPE_CENTER[0] + r * np.cos(th), COUPE_CENTER[1] + r * np.sin(th))
                if all(np.hypot(p[0] - q[0], p[1] - q[1]) > 0.021 for q in pts):
                    pts.append(p)
                    break
            else:
                raise RuntimeError("coupe sampling failed")
        return pts[len(existing):]

    def coupe_crop(env, frame_hi, scale=1.5):
        from gyoza.envs.place_skill import project_bowl_bbox
        x0, y0, x1, y1 = project_bowl_bbox(env, COUPE_CENTER, r=0.075)
        s = scale
        return frame_hi[int(y0 * s):int(y1 * s), int(x0 * s):int(x1 * s)]

    labels = []
    for i in range(n_pairs):
        query = QUERY_CYCLE[i % len(QUERY_CYCLE)]
        base = {"shiratama": int(rng.integers(0, 3)),
                "grape": int(rng.integers(0, 3)),
                "cherry_stage3_red": int(rng.integers(0, 2))}
        added = bool(rng.random() < 0.5)

        # クープ内容を statics として設置（測定と同一の 4-tuple z 指定）
        contents = []
        for fruit, cnt in base.items():
            for p in sample_coupe_positions(cnt, [c[1] for c in contents]):
                contents.append((fruit, p, float(rng.uniform(0, 2 * np.pi))))
        statics = [(nm, xy, yaw, COUPE_FLOOR_Z) for nm, xy, yaw in contents]

        os.environ["GYOZA_TOMATO"] = query
        env = PickPlaceEnv(seed=int(rng.integers(1 << 31)), render_obs=True,
                           statics=statics, statics_group=5)
        env.reset()
        tq = env._tomato_qpos

        def teleport(xy, z):
            env.data.qpos[tq:tq + 7] = [xy[0], xy[1], z, 1, 0, 0, 0]
            env.data.qvel[:] = 0
            mujoco.mj_forward(env.model, env.data)
            for _ in range(80):
                mujoco.mj_step(env.model, env.data)

        hi_r = mujoco.Renderer(env.model, 720, 960)
        full_opt = mujoco.MjvOption()
        full_opt.geomgroup[5] = 1

        def render_hi():
            hi_r.update_scene(env.data, camera="overhead", scene_option=full_opt)
            return hi_r.render()

        # before: ライブ果物は盤上（クロップ外）
        teleport((-0.03, -0.10), BOARD_Z + VEG_HEIGHTS[query] / 2 + 0.005)
        before = coupe_crop(env, render_hi())

        if added:
            p = sample_coupe_positions(1, [c[1] for c in contents])[0]
            teleport(p, COUPE_FLOOR_Z + VEG_HEIGHTS[query] / 2 + 0.005)
        else:
            teleport((0.05, -0.04), BOARD_Z + VEG_HEIGHTS[query] / 2 + 0.005)
        after = coupe_crop(env, render_hi())

        pid = f"p{i:03d}"
        imageio.imwrite(out / f"{pid}_before.png", before)
        imageio.imwrite(out / f"{pid}_after.png", after)
        labels.append(dict(id=pid, fruit=query, base=base, added=added))
        hi_r.close()
        env.close()
        if (i + 1) % 20 == 0:
            print(f"[bench] {i + 1}/{n_pairs}", flush=True)

    out.mkdir(parents=True, exist_ok=True)   # bucket FUSE 空 dir バグ対策
    (out / "labels.json").write_text(json.dumps(labels, indent=1))
    n_pos = sum(l["added"] for l in labels)
    print(f"[bench] DONE pairs={n_pairs} pos={n_pos} neg={n_pairs - n_pos}", flush=True)


if __name__ == "__main__":
    main()
