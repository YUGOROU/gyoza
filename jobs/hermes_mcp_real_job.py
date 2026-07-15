# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = ["hermes-agent", "fastmcp", "httpx", "mujoco==3.10.0", "numpy", "gymnasium", "stable-baselines3", "torch", "imageio[ffmpeg]", "huggingface_hub>=0.34"]
# ///
"""AIAgent を GYOZA MCP サーバ（実 SimRunner・MuJoCo）経由で駆動する e2e 検証（GPU 隔離）。

hermes_mcp_drive_job.py（モック）の実 sim 版。MCP ツールの中身を space/gyoza_mcp.py の実
SimRunner に差し替え、GLM-5.2（HF router）が pick_place→check_in_bowl→retry→次、を実際の
MuJoCo 盛り付けで駆動できるかを見る。成功で「AIAgent+MCP+実sim」構造が確定 → Space 配線へ。

MCP サーバは hermes discovery が stdio サブプロセスとして起動する。サブプロセスの環境
（MUJOCO_GL/RL_MODEL/PYTHONPATH）を確実にするため bash -c ラッパで env を明示注入する。

実行:
    hf jobs uv run --flavor t4-small --timeout 40m --detach --secrets HF_TOKEN \
        -v hf://buckets/YUGOROU/gyoza-sim:/gyoza --env GYOZA_DATA=/gyoza \
        --env MODEL=zai-org/GLM-5.2 --env RL_MODEL_REL=outputs/rl_v3a-warm-perf/best/best_model.zip \
        jobs/hermes_mcp_real_job.py
"""
import logging
import os
import pathlib
import subprocess
import sys
import traceback

logging.basicConfig(level=logging.WARNING)
LOG = logging.getLogger("gyoza-real")
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
    gl = pick_gl_backend()
    token = os.environ.get("HF_TOKEN")
    model = os.environ.get("MODEL", "zai-org/GLM-5.2")
    rl_rel = os.environ.get("RL_MODEL_REL", "outputs/rl_v3a-warm-perf/best/best_model.zip")
    rl_abs = str(DATA / rl_rel)

    import hermes_constants
    from hermes_cli.config import load_config, save_config

    home = pathlib.Path(hermes_constants.get_hermes_home())
    home.mkdir(parents=True, exist_ok=True)

    # MCP サーバ（実 SimRunner）を stdio 起動。サブプロセス env を bash -c で確実に注入。
    server = str(CODE / "space" / "gyoza_mcp.py")
    envline = (f"MUJOCO_GL={gl} PYOPENGL_PLATFORM={gl} RL_MODEL={rl_abs} "
               f"GYOZA_BOWL_CENTER=0.204,-0.098 HF_TOKEN={token} "
               f"PYTHONPATH={CODE}:{CODE}/space")
    cfg = load_config() or {}
    cfg.setdefault("_config_version", 9)
    cfg.setdefault("mcp_servers", {})["gyoza"] = {
        "command": "bash", "args": ["-c", f"{envline} exec python {server}"]}
    save_config(cfg)
    print(f"[config] gyoza MCP -> bash -c '{envline[:60]}... python {server}'", flush=True)

    from hermes_cli.mcp_startup import start_background_mcp_discovery, wait_for_mcp_discovery
    start_background_mcp_discovery(logger=LOG, thread_name="gyoza-mcp-disc")
    wait_for_mcp_discovery(90)   # SimRunner ロード込みで余裕を持たせる

    try:
        from model_tools import get_tool_definitions
        defs = get_tool_definitions()
        names = [d.get("function", {}).get("name", d.get("name", "?"))
                 for d in defs] if isinstance(defs, list) else list(defs)
        gyoza_tools = [n for n in names if "gyoza" in str(n)]
        print(f"[tools] total={len(names)} gyoza={gyoza_tools}", flush=True)
    except Exception:  # noqa: BLE001
        traceback.print_exc()

    from run_agent import AIAgent

    trace = []

    def on_tool_start(*a, **k):
        print(f"[tool_start] {a[:2] if len(a) >= 2 else a} args={a[2] if len(a) > 2 else ''}", flush=True)
        trace.append(("start", a))

    def on_tool_complete(*a, **k):
        res = str(a[3])[:140] if len(a) > 3 else ""
        print(f"[tool_complete] {a[1] if len(a) > 1 else ''} -> {res}", flush=True)
        trace.append(("complete", a))

    agent = AIAgent(
        base_url="https://router.huggingface.co/v1", api_key=token, model=model,
        max_iterations=24, quiet_mode=True, skip_memory=True, skip_context_files=True,
        tool_start_callback=on_tool_start, tool_complete_callback=on_tool_complete)
    prompt = (
        "You are plating a ramen bowl with a robot. Place toppings one at a time by calling the "
        "pick_place tool, then verify each by calling check_in_bowl. If check_in_bowl returns "
        "'no', retry pick_place for the same topping (up to 2 retries), then move on. Place naruto "
        "first at goal (0.204,-0.096), then ajitama at (0.20,-0.10). Reply DONE when finished.")
    print(f"[agent] model={model} run_conversation (real sim)...", flush=True)
    try:
        resp = agent.run_conversation(prompt)
        fr = resp.get("final_response") if isinstance(resp, dict) else resp
        print("[agent] final:", str(fr)[:400], flush=True)
    except Exception:  # noqa: BLE001
        traceback.print_exc()

    starts = [t for t in trace if t[0] == "start"]
    print(f"=== DONE real drive: tool_start={len(starts)} ===", flush=True)


if __name__ == "__main__":
    main()
