# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = ["hermes-agent", "fastmcp", "httpx"]
# ///
"""AIAgent（hermes-agent）を GYOZA MCP サーバ経由で駆動する構造実証（隔離 CPU）。

D7 の正しい構造の end-to-end 疎通確認: 手書きループを書かず、hermes-agent の AIAgent を
ライブラリ import し、GYOZA sim ツールを FastMCP stdio サーバ（ここではモック＝MuJoCo 非依存）
として ~/.hermes/config.yaml の mcp_servers に登録 → MCP discovery → AIAgent が MCP 経由で
pick_place/check_in_bowl を呼び、盛り付けループ（配置→判定→retry→次）を駆動できるかを見る。

成功すれば「AIAgent+MCP+skill」構造が成立 → 次は MCP ツールの中身を実 SimRunner に差し替え、
Space に配線（tool_start/complete/event コールバックで UI stream）。

モデルは HF router（OpenAI 互換）: base_url=https://router.huggingface.co/v1, api_key=HF_TOKEN。
既定モデルは GLM-5.2（tool-call 堅牢）。MODEL env で差し替え可。

実行:
    hf jobs uv run --flavor cpu-upgrade --timeout 30m --detach --secrets HF_TOKEN \
        --env MODEL=zai-org/GLM-5.2 jobs/hermes_mcp_drive_job.py
"""
import logging
import os
import sys
import textwrap
import traceback
from pathlib import Path

logging.basicConfig(level=logging.WARNING)
LOG = logging.getLogger("gyoza-drive")


GYOZA_MCP_SRC = '''\
"""GYOZA 盛り付けツールの MCP サーバ（構造実証用モック・MuJoCo 非依存）。
実配線では中身を space/sim_runner.py の SimRunner（place/judge_added/render）に差し替える。"""
from fastmcp import FastMCP

mcp = FastMCP("gyoza")
_state = {"naruto_attempts": 0}


@mcp.tool
def pick_place(object: str, goal_x: float, goal_y: float) -> str:
    """Grasp a ramen topping and place it at a goal point in the bowl.
    Reachable zone: small disk (~1.8cm) around bowl center (0.204,-0.096) in table meters."""
    # naruto は初回だけ丼外に落ちる（retry を誘発）。他は成功。
    if object == "naruto":
        _state["naruto_attempts"] += 1
        landed = _state["naruto_attempts"] >= 2
    else:
        landed = True
    where = "inside the bowl" if landed else "on the rim, NOT inside the bowl"
    return f"pick_place({object}) executed at ({goal_x:.3f},{goal_y:.3f}); it landed {where}."


@mcp.tool
def check_in_bowl(object: str) -> str:
    """Visually verify whether the given topping landed inside the bowl in the latest attempt."""
    if object == "naruto":
        ok = _state["naruto_attempts"] >= 2
    else:
        ok = True
    return "yes" if ok else "no"


if __name__ == "__main__":
    mcp.run()
'''


def main():
    token = os.environ.get("HF_TOKEN")
    model = os.environ.get("MODEL", "zai-org/GLM-5.2")
    if not token:
        print("ERROR: HF_TOKEN not set", file=sys.stderr); sys.exit(1)

    import hermes_constants
    from hermes_cli.config import load_config, save_config

    home = Path(hermes_constants.get_hermes_home())
    home.mkdir(parents=True, exist_ok=True)
    print(f"[home] {home}", flush=True)

    # GYOZA MCP サーバを書き出し、config.yaml の mcp_servers に stdio 登録
    mcp_path = home / "gyoza_mcp.py"
    mcp_path.write_text(GYOZA_MCP_SRC)
    cfg = load_config() or {}
    cfg.setdefault("_config_version", 9)
    cfg.setdefault("mcp_servers", {})["gyoza"] = {
        "command": sys.executable, "args": [str(mcp_path)]}
    save_config(cfg)
    print(f"[config] mcp_servers.gyoza -> {sys.executable} {mcp_path}", flush=True)

    # MCP discovery（バックグラウンドスレッド → 待機）
    from hermes_cli.mcp_startup import start_background_mcp_discovery, wait_for_mcp_discovery
    start_background_mcp_discovery(logger=LOG, thread_name="gyoza-mcp-disc")
    wait_for_mcp_discovery(45)

    # 発見されたツール一覧（GYOZA の pick_place/check_in_bowl が入っているか）
    try:
        from model_tools import get_tool_definitions
        defs = get_tool_definitions()
        names = [d.get("function", {}).get("name", d.get("name", "?"))
                 for d in defs] if isinstance(defs, list) else list(defs)
        gyoza_tools = [n for n in names if "pick_place" in str(n) or "check_in_bowl" in str(n)]
        print(f"[tools] total={len(names)} gyoza={gyoza_tools}", flush=True)
    except Exception:  # noqa: BLE001
        print("[tools] get_tool_definitions introspection failed:", flush=True)
        traceback.print_exc()

    # AIAgent を構築（HF router 直指定）+ ツールコール stream コールバック
    from run_agent import AIAgent

    trace = []

    def on_tool_start(*a, **k):
        print(f"[tool_start] args={a} kwargs={ {kk: str(vv)[:80] for kk, vv in k.items()} }", flush=True)
        trace.append(("start", a, k))

    def on_tool_complete(*a, **k):
        print(f"[tool_complete] args={a} kwargs={ {kk: str(vv)[:80] for kk, vv in k.items()} }", flush=True)
        trace.append(("complete", a, k))

    agent = AIAgent(
        base_url="https://router.huggingface.co/v1",
        api_key=token,
        model=model,
        max_iterations=20,
        quiet_mode=True,
        skip_memory=True,
        skip_context_files=True,
        tool_start_callback=on_tool_start,
        tool_complete_callback=on_tool_complete,
    )
    prompt = textwrap.dedent("""\
        You are plating a ramen bowl using a robot. Place toppings one at a time by calling the
        pick_place tool, then verify each with check_in_bowl. If check_in_bowl says "no", retry
        pick_place for the same topping (up to 2 retries). Place naruto first at goal
        (0.204,-0.096), then ajitama at (0.20,-0.10). When both are verified in the bowl, reply
        DONE.""")
    print(f"[agent] model={model} run_conversation...", flush=True)
    try:
        resp = agent.run_conversation(prompt)
        print("[agent] final response:", str(resp)[:500], flush=True)
    except Exception:  # noqa: BLE001
        print("[agent] run_conversation raised:", flush=True)
        traceback.print_exc()

    calls = [t for t in trace if t[0] == "start"]
    print(f"=== DONE drive: tool_start events={len(calls)} ===", flush=True)


if __name__ == "__main__":
    main()
