"""VLA ゼロショット pick_place 成功率測定（実験計画 §4 手順1 / ゲート G2 の入力）。

sim (PickPlaceEnv) の obs を SO101Adapter 契約（overhead/side 2フレーム + state6 度）で
VLA に食わせ、6次元 absolute 関節目標（度）を follower 腕へ指令するロールアウトを
N エピソード実行し、成功率（GT 幾何判定）を JSON に書き出す。

実行例:
    python scripts/zeroshot_eval.py --model smolvla --episodes 5 --device mps   # ローカル疎通
    python scripts/zeroshot_eval.py --model smolvla --episodes 50               # HF Jobs (CUDA)

モデル追加は MODEL_SPECS に1エントリ（VLA-Bench modal_zeroshot_smoke.py と同形式）。
molmoact2 は lerobot 系と別系統（gyoza/vla/molmoact_adapter.py、raw スケール + action chunk 実行）。
"""

from __future__ import annotations

import argparse
import importlib
import json
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

MODEL_SPECS = {
    "smolvla": {
        "repo": "lerobot/smolvla_base",
        "import_path": "lerobot.policies.smolvla.modeling_smolvla",
        "class_name": "SmolVLAPolicy",
    },
    "pi05": {
        "repo": "lerobot/pi05_base",
        "import_path": "lerobot.policies.pi05.modeling_pi05",
        "class_name": "PI05Policy",
    },
    "xvla": {
        "repo": "lerobot/xvla-base",
        "import_path": "lerobot.policies.xvla.modeling_xvla",
        "class_name": "XVLAPolicy",
    },
    "molmoact2": {"repo": "allenai/MolmoAct2-SO100_101", "kind": "molmoact"},
}

TASK = "Pick up the red tomato and place it on the white plate."


class _LerobotRunner:
    """SO101Adapter（1 ステップ 1 推論・policy 内部の action queue 任せ）。"""

    def __init__(self, policy, adapter):
        self.policy, self.adapter = policy, adapter

    def reset(self):
        self.policy.reset()

    def predict_chunk(self, frames, state, task):
        return self.adapter.predict(frames, state, task=task).reshape(1, 6)


class _MolmoRunner:
    """MolmoActAdapter（action chunk (30,6) をまとめて実行してから再推論）。"""

    def __init__(self, adapter):
        self.adapter = adapter

    def reset(self):
        pass

    def predict_chunk(self, frames, state, task):
        return self.adapter.predict_chunk(frames, state, task)


def load_runner(model_name: str, device: str, stats_repo: str | None):
    spec = MODEL_SPECS[model_name]
    if spec.get("kind") == "molmoact":
        from gyoza.vla.molmoact_adapter import MolmoActAdapter
        return _MolmoRunner(MolmoActAdapter(device=device))

    from gyoza.vla.so101_vla_adapter import SO101Adapter
    mod = importlib.import_module(spec["import_path"])
    policy = getattr(mod, spec["class_name"]).from_pretrained(spec["repo"]).to(device).eval()
    adapter = SO101Adapter.from_policy(policy, spec["repo"], device, stats_repo=stats_repo)
    return _LerobotRunner(policy, adapter)


def run_episode(env, runner, max_steps: int, video_frames: list | None):
    obs = env.reset()
    runner.reset()
    t = 0
    while t < max_steps:
        chunk = runner.predict_chunk(obs["frames"], obs["state"], TASK)
        for action in chunk:
            obs = env.step(action)
            t += 1
            if video_frames is not None:
                video_frames.append(obs["frames"]["overhead"])
            if env.check_success():
                return True, t
            if t >= max_steps:
                break
    return False, max_steps


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="smolvla", choices=sorted(MODEL_SPECS))
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--max-steps", type=int, default=300, help="30Hz制御で10s相当")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None, help="cuda/mps/cpu（既定: 自動）")
    p.add_argument("--stats-repo", default="YUGOROU/act_grasp_almond",
                   help="'none' で合成 stats")
    p.add_argument("--out", default="outputs/zeroshot")
    p.add_argument("--video-every", type=int, default=10, help="k エピソードごとに動画保存（0=無効）")
    args = p.parse_args()

    import torch
    device = args.device or (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu")
    stats_repo = None if args.stats_repo == "none" else args.stats_repo

    out_dir = pathlib.Path(args.out) / f"{args.model}_seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[eval] model={args.model} episodes={args.episodes} device={device}")
    t0 = time.time()
    runner = load_runner(args.model, device, stats_repo)
    print(f"[eval] policy loaded in {time.time() - t0:.1f}s")

    from gyoza.envs.pick_place import PickPlaceEnv
    env = PickPlaceEnv(seed=args.seed)

    results = []
    for ep in range(args.episodes):
        frames = [] if args.video_every and ep % args.video_every == 0 else None
        t0 = time.time()
        success, steps = run_episode(env, runner, args.max_steps, frames)
        dt = time.time() - t0
        results.append({"episode": ep, "success": success, "steps": steps, "wall_s": round(dt, 1)})
        n_ok = sum(r["success"] for r in results)
        print(f"[ep {ep:03d}] success={success} steps={steps} {dt:.1f}s "
              f"| running rate {n_ok}/{ep + 1} = {n_ok / (ep + 1):.1%}", flush=True)
        if frames:
            import imageio
            imageio.mimsave(out_dir / f"ep{ep:03d}_{'ok' if success else 'ng'}.mp4",
                            frames, fps=30)

    rate = sum(r["success"] for r in results) / len(results)
    summary = {
        "model": args.model, "repo": MODEL_SPECS[args.model]["repo"],
        "episodes": args.episodes, "max_steps": args.max_steps, "seed": args.seed,
        "device": device, "stats_repo": stats_repo, "task": TASK,
        "success_rate": rate, "results": results,
    }
    (out_dir / "results.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\n=== {args.model} zero-shot success rate: {rate:.1%} "
          f"({sum(r['success'] for r in results)}/{len(results)}) ===")
    print(f"[eval] wrote {out_dir / 'results.json'}")
    print(f"[G2] {'合成データ採用ライン(≥20%)到達' if rate >= 0.20 else '20%未満 → テレオペ分岐の検討対象'}")


if __name__ == "__main__":
    main()
