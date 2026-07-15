# /// script
# requires-python = ">=3.11"
# dependencies = ["mujoco", "numpy", "imageio[ffmpeg]", "huggingface_hub", "pillow"]
# ///
"""G1 ゲート: HF Jobs 上で MuJoCo ヘッドレス描画（EGL、NG なら osmesa）+ 双腕 SO-101 疎通。

【PASS 済 2026-07-05 @ t4-small: MUJOCO_GL=egl, 194 env-steps/s】環境更新時の再検証用に維持。

bucket YUGOROU/gyoza-sim を /gyoza に rw マウント（/data はローカル uv script 実行時に予約済で不可）。

実行:
    hf jobs uv run --flavor t4-small --timeout 30m --secrets HF_TOKEN \
        -v hf://buckets/YUGOROU/gyoza-sim:/gyoza --env GYOZA_DATA=/gyoza jobs/g1_smoke.py
"""

import json
import os
import pathlib
import subprocess
import sys
import time

DATA = pathlib.Path(os.environ.get("GYOZA_DATA", "/data"))
CODE = DATA / "code"


def sh(cmd: str) -> str:
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return (r.stdout + r.stderr).strip()


def try_backend(backend: str) -> bool:
    """サブプロセスで MUJOCO_GL=backend の描画を試す（プロセス単位でしか切替不可）。"""
    code = (
        "import mujoco, numpy as np;"
        "m = mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom size=\"0.1\"/></worldbody></mujoco>');"
        "d = mujoco.MjData(m); mujoco.mj_forward(m, d);"
        "r = mujoco.Renderer(m, 240, 320); r.update_scene(d);"
        "img = r.render(); print('render mean', float(img.mean()))"
    )
    env = dict(os.environ, MUJOCO_GL=backend)
    r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=120)
    ok = r.returncode == 0
    print(f"[gl] backend={backend}: {'OK' if ok else 'NG'}"
          + ("" if ok else f"\n{(r.stderr or '')[-500:]}"))
    return ok


def main():
    print("[info]", sh("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader") or "no GPU")
    assert CODE.is_dir(), f"{CODE} が無い（bucket を -v hf://buckets/YUGOROU/gyoza-sim:/gyoza でマウントし GYOZA_DATA=/gyoza を渡したか?）"
    sh("apt-get update -qq && apt-get install -y -qq libegl1 libgles2 libosmesa6 ffmpeg")

    backend = next((b for b in ("egl", "osmesa") if try_backend(b)), None)
    if backend is None:
        print("[G1] FAIL: どのバックエンドでも描画不可")
        sys.exit(1)
    os.environ["MUJOCO_GL"] = backend
    print(f"[G1] using MUJOCO_GL={backend}")

    sys.path.insert(0, str(CODE))
    import numpy as np
    from gyoza.envs.pick_place import PickPlaceEnv

    env = PickPlaceEnv(seed=0)
    obs = env.reset()
    print("[env] reset OK  state(deg) =", obs["state"].round(1).tolist())

    # follower 腕をサイン波で駆動しつつ両カメラ描画（90 制御ステップ = 3s 相当）
    frames_o, frames_s = [], []
    home_deg = obs["state"].astype(float)
    t0 = time.time()
    for t in range(90):
        target = home_deg + np.array([25, 15, -15, 10, 0, 30]) * np.sin(t / 15.0)
        obs = env.step(target)
        frames_o.append(obs["frames"]["overhead"])
        frames_s.append(obs["frames"]["side"])
    dt = time.time() - t0
    fps = 90 / dt
    print(f"[perf] 90 steps (physics + 2cam 640x480 render) in {dt:.1f}s = {fps:.1f} env-steps/s")

    import imageio
    out = DATA / "outputs" / "g1"
    out.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out / "g1_overhead.mp4", frames_o, fps=30)
    imageio.mimsave(out / "g1_side.mp4", frames_s, fps=30)
    summary = {
        "gate": "G1", "backend": backend, "env_steps_per_s": round(fps, 1),
        "gpu": sh("nvidia-smi --query-gpu=name --format=csv,noheader"),
        "success_check": env.check_success(), "state_final": obs["state"].round(1).tolist(),
    }
    (out / "g1_summary.json").write_text(json.dumps(summary, indent=2))
    print("[G1] PASS —", json.dumps(summary))


if __name__ == "__main__":
    main()
