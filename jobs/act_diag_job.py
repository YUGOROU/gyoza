# /// script
# requires-python = ">=3.11"
# dependencies = ["mujoco", "numpy", "lerobot==0.4.4", "torch", "torchvision", "imageio[ffmpeg]", "huggingface_hub"]
# ///
"""ACT 0% の切り分け診断ジョブ（並列A）。

eval_act_synth の症状: 接近初動は正しいがその後凍結（同一姿勢に張り付き）。
仮説を2つに切り分ける:
  A) teacher forcing: 学習データのフレームを順に入力し予測アクションと GT を比較。
     ここが悪ければモデル/入力パイプラインの不整合（正規化・キー名・単位）。
  B) 再計画頻度: n_action_steps=100（既定、3.3s 開ループ）→10 で閉ループ再評価。
     A が良好で B で成功が出るなら「チャンク境界の分布外落ち」が原因。

実行:
    hf jobs uv run --flavor t4-small --timeout 2h --secrets HF_TOKEN \
        -v hf://buckets/YUGOROU/gyoza-sim:/gyoza --env GYOZA_DATA=/gyoza \
        --env POLICY=YUGOROU/act_gyoza_pickplace_synth \
        --env DATASET=YUGOROU/gyoza-pickplace-synth \
        --env RUN=diag_act_synth jobs/act_diag_job.py
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
        if r.returncode == 0:
            print(f"[gl] {backend}", flush=True)
            return backend
    raise RuntimeError("no GL backend")


def main():
    subprocess.run("apt-get update -qq && apt-get install -y -qq libegl1 libgles2 libosmesa6 ffmpeg",
                   shell=True, capture_output=True)
    os.environ["MUJOCO_GL"] = pick_gl_backend()
    sys.path.insert(0, str(CODE))

    import numpy as np
    import torch

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.act.modeling_act import ACTPolicy
    from gyoza.envs.pick_place import PickPlaceEnv

    policy_repo = os.environ["POLICY"]
    dataset_repo = os.environ["DATASET"]
    run = os.environ.get("RUN", "diag_act")
    out = DATA / "outputs" / "evals" / run
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    policy = ACTPolicy.from_pretrained(policy_repo)
    policy.to(device).eval()
    print(f"[diag] config: chunk={policy.config.chunk_size} n_action_steps={policy.config.n_action_steps}",
          flush=True)
    print(f"[diag] input_features: {list(policy.config.input_features)}", flush=True)

    # ---- A) teacher forcing on training episode 0 ----
    ds = LeRobotDataset(dataset_repo, episodes=[0])
    n = ds.num_frames
    print(f"[diag] dataset ep0 frames={n}", flush=True)
    policy.reset()
    errs, preds, gts = [], [], []
    for i in range(n):
        item = ds[i]
        batch = {
            "observation.images.overhead": item["observation.images.overhead"].unsqueeze(0).to(device),
            "observation.state": item["observation.state"].unsqueeze(0).to(device),
            "task": [TASK],
        }
        with torch.inference_mode():
            a = policy.select_action(batch).squeeze(0).cpu().numpy()
        gt = item["action"].numpy()
        preds.append(a)
        gts.append(gt)
        errs.append(np.abs(a - gt))
    errs = np.array(errs)
    print(f"[diag A] MAE per joint (deg): {np.round(errs.mean(axis=0), 3).tolist()}", flush=True)
    print(f"[diag A] MAE overall: {errs.mean():.3f} deg, p95: {np.percentile(errs, 95):.3f}", flush=True)
    # 予測が時間方向に変化しているか（凍結の有無）
    pv = np.array(preds)
    print(f"[diag A] pred action std over time: {np.round(pv.std(axis=0), 3).tolist()}", flush=True)

    # ---- B) 閉ループ n_action_steps=10 で 10 エピソード ----
    policy.config.n_action_steps = 10
    env = PickPlaceEnv(seed=42, render_obs=True)
    succ = 0
    for ep in range(10):
        env.reset()
        policy.reset()
        done = False
        for t in range(400):
            img = env.render("overhead")
            batch = {
                "observation.images.overhead":
                    torch.from_numpy(img).permute(2, 0, 1).float().div(255).unsqueeze(0).to(device),
                "observation.state":
                    torch.from_numpy(env.state_deg().astype(np.float32)).unsqueeze(0).to(device),
                "task": [TASK],
            }
            with torch.inference_mode():
                a = policy.select_action(batch).squeeze(0).cpu().numpy()
            env.step(a.astype(np.float64))
            if env.check_success():
                done = True
                break
        succ += done
        print(f"[diag B] ep{ep} success={done} steps={t + 1}", flush=True)
    print(f"[diag B] n_action_steps=10 -> success {succ}/10", flush=True)

    (out / "diag.json").write_text(json.dumps(dict(
        mae=float(errs.mean()), succ_nas10=succ), indent=2))
    print("[diag] DONE", flush=True)


if __name__ == "__main__":
    main()
