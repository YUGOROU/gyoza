# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = ["hermes-agent", "fastmcp", "httpx"]
# ///
"""hermes-agent 導入 + AIAgent 埋め込み可否のクラウド診断（隔離 CPU コンテナ）。

正しい構造（D7）: orchestrator は手書きループでなく hermes-agent の AIAgent（run_agent.py）を
ライブラリとして import して駆動し、GYOZA sim ツールは MCP サーバとして食わせる。その第一関門
＝「素の pip install で `from run_agent import AIAgent` が通るか」を隔離コンテナで確認する
（run_agent は import 時に tools.terminal_tool / tools.browser_tool 等を読むため要検証）。

段階診断（各段 try/except で切り分け、どこで落ちるかを可視化）:
  S1: import 経路（run_agent / hermes_constants / model_tools / tools.mcp_tool / fastmcp）
  S2: AIAgent のコンストラクタ introspection（MCP/コールバック引数の確認）
  S3: MCP 登録機構（hermes_cli.mcp_config / tools.mcp_tool の API 表層）

実行:
    hf jobs uv run --flavor cpu-upgrade --timeout 30m --detach --secrets HF_TOKEN \
        jobs/hermes_mcp_spike_job.py
"""
import importlib
import inspect
import sys
import traceback


def probe(label, fn):
    try:
        val = fn()
        print(f"[OK ] {label}: {val}", flush=True)
        return val
    except Exception as e:  # noqa: BLE001
        print(f"[NG ] {label}: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
        return None


def main():
    print(f"python {sys.version}", flush=True)

    # --- S1: import 経路 ---
    def _imp(mod):
        m = importlib.import_module(mod)
        return getattr(m, "__file__", "?")
    probe("import hermes_constants", lambda: _imp("hermes_constants"))
    probe("import fastmcp", lambda: _imp("fastmcp"))
    probe("import model_tools", lambda: _imp("model_tools"))
    probe("import tools.mcp_tool", lambda: _imp("tools.mcp_tool"))
    run_agent = probe("import run_agent", lambda: importlib.import_module("run_agent"))
    if run_agent is None:
        print("=== AIAgent import 不可 → 素の pip install では埋め込めない。"
              "browser/terminal tool 等の欠落を上の traceback で確認し、"
              "必要 extras / 環境変数 / システム依存を洗い出す ===", flush=True)
        return

    AIAgent = probe("getattr AIAgent", lambda: run_agent.AIAgent)
    if AIAgent is None:
        return

    # --- S2: AIAgent constructor introspection ---
    def _sig():
        sig = inspect.signature(AIAgent.__init__)
        keys = list(sig.parameters.keys())
        mcp_like = [k for k in keys if "mcp" in k.lower() or "tool" in k.lower()
                    or "callback" in k.lower()]
        return f"{len(keys)} params; mcp/tool/callback: {mcp_like}"
    probe("AIAgent.__init__ signature", _sig)

    # --- S3: MCP 登録機構の表層 ---
    def _mcp_config_api():
        m = importlib.import_module("hermes_cli.mcp_config")
        return [n for n in dir(m) if not n.startswith("_")][:25]
    probe("hermes_cli.mcp_config public API", _mcp_config_api)

    def _mcp_tool_api():
        m = importlib.import_module("tools.mcp_tool")
        return [n for n in dir(m) if not n.startswith("_")][:25]
    probe("tools.mcp_tool public API", _mcp_tool_api)

    # AIAgent が MCP をどこから読むか（config.yaml mcp_servers / startup）
    probe("import hermes_cli.mcp_startup", lambda: _imp("hermes_cli.mcp_startup"))

    print("=== DONE spike S1-S3 ===", flush=True)


if __name__ == "__main__":
    main()
