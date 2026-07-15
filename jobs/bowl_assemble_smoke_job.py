# /// script
# requires-python = ">=3.11"
# dependencies = ["mujoco==3.10.0", "numpy", "gymnasium", "stable-baselines3", "torch", "imageio[ffmpeg]", "huggingface_hub>=0.34"]
# ///
"""盛り付け積み上げの end-to-end スモーク（クラウド）— SimRunner の検証。

事前配置チャーシュー/メンマ + naruto/味玉を place→VLM判定(Kimi-K2.6)→retry で積み上げ、
連続動画 + 判定ログを bucket に保存。エージェント役は当面スクリプト（hermes-agent 配線前の
sim+ツール+VLM の疎通確認）。重い sim はローカルでなくクラウドで回す方針に従う。

実行:
    hf jobs uv run --flavor t4-small --timeout 1h --secrets HF_TOKEN \
        -v hf://buckets/YUGOROU/gyoza-sim:/gyoza --env GYOZA_DATA=/gyoza \
        --env MODEL=outputs/rl_v3a-warm-perf/best/best_model.zip \
        --env RUN=assemble_smoke --env VLM=moonshotai/Kimi-K2.6 \
        jobs/bowl_assemble_smoke_job.py
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
    sys.path.insert(0, str(CODE))                 # gyoza パッケージ
    sys.path.insert(0, str(CODE / "space"))       # sim_runner

    import imageio
    from sim_runner import SimRunner

    run = os.environ.get("RUN", "assemble_smoke")
    model_rel = os.environ["MODEL"]
    vlm = os.environ.get("VLM", "moonshotai/Kimi-K2.6")
    out = DATA / "outputs" / "assemble" / run
    out.mkdir(parents=True, exist_ok=True)

    cx, cy = 0.204, -0.098
    pre = [("chashu", (cx - 0.016, cy - 0.012), 0.3), ("menma", (cx - 0.004, cy + 0.018), -0.6)]
    plan = [("naruto", (0.210, -0.090), "pink white spiral fish cake"),
            ("ajitama", (0.198, -0.100), "halved soft-boiled egg")]

    runner = SimRunner(str(DATA / model_rel), bowl_center=(cx, cy), soup_top=0.030,
                       pre_placed=pre, vlm_model=vlm)
    frames, log = [], []
    for topping, goal, desc in plan:
        for attempt in range(3):
            print(f"[agent] pick_place({topping}, {goal}) attempt {attempt}", flush=True)
            res = runner.place(topping, goal, release_tilt=20.0, seed=attempt * 7 + 1,
                               on_frame=frames.append)
            j = runner.judge_added(res["before_crop_b64"], res["after_crop_b64"], topping, desc)
            rec = dict(topping=topping, attempt=attempt, gt=res["success_gt"],
                       landing=[round(v, 4) for v in res["landing_xy"]],
                       vlm=j["verdict"], raw=j["raw"][:60])
            log.append(rec)
            print(f"[agent] {rec}", flush=True)
            if j["verdict"]:
                break

    imageio.imwrite(out / "final.png", runner.render_scene())
    imageio.mimsave(out / "assemble.mp4", frames[::2], fps=30)
    (out / "log.json").write_text(json.dumps(dict(placed=[p[0] for p in runner.placed],
                                                  steps=log), indent=2))
    print("[assemble] DONE", json.dumps(dict(placed=[p[0] for p in runner.placed],
                                             n_frames=len(frames))), flush=True)


if __name__ == "__main__":
    main()
