"""GYOZA 盛り付けデモ Space — FastAPI + websocket ライブUI。

エージェント（hermes-agent, 別配線）が MCP 経由で SimRunner ツールを叩く構成が本命だが、
本 app は「sim バックエンド + ライブUI」を提供する土台。websocket で:
  - {"type":"frame", "data": <jpeg b64>}         俯瞰ライブプレビュー
  - {"type":"trace", "role":..., "text":...}      エージェント/ツールのトレース
  - {"type":"verdict", "topping":..., "ok":bool}  postcondition 判定
  - {"type":"done", ...}

/ws に接続すると assemble が走る。当面はスクリプト driver（run_assembly）で逐次+retry を
駆動しUI配線を検証する（hermes-agent 配線後は agent のトレースがこの emit を置き換える）。
"""
from __future__ import annotations

import asyncio
import base64
import io
import os
import pathlib
import threading

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

HERE = pathlib.Path(__file__).resolve().parent
MODEL_PATH = os.environ.get("RL_MODEL", str(HERE / "best_model.zip"))

CX, CY = 0.204, -0.098
PRE_PLACED = [("chashu", (CX - 0.016, CY - 0.012), 0.3), ("menma", (CX - 0.004, CY + 0.018), -0.6)]
# ライブ配置プラン（トッピング, ゴール絶対xy, 説明）
PLAN = [("naruto", (0.210, -0.090), "pink white spiral fish cake"),
        ("ajitama", (0.198, -0.100), "halved soft-boiled egg")]

app = FastAPI()
_runner = None
_runner_lock = threading.Lock()


def get_runner():
    global _runner
    with _runner_lock:
        if _runner is None:
            import sys
            sys.path.insert(0, str(HERE.parent))  # gyoza パッケージ
            from sim_runner import SimRunner
            _runner = SimRunner(MODEL_PATH, bowl_center=(CX, CY), soup_top=0.030,
                                pre_placed=PRE_PLACED)
    return _runner


def _jpeg_b64(rgb) -> str:
    import imageio
    buf = io.BytesIO()
    imageio.imwrite(buf, rgb, format="jpeg", quality=80)
    return base64.b64encode(buf.getvalue()).decode()


def run_assembly(emit):
    """スクリプト driver（UI配線検証用・hermes-agent がここを置換）。emit(dict) で UI へ。"""
    runner = get_runner()
    emit({"type": "trace", "role": "system", "text": "丼にチャーシュー・メンマを配置済み。盛り付けを開始します。"})
    emit({"type": "frame", "data": _jpeg_b64(runner.render_scene())})
    for topping, goal, desc in PLAN:
        placed_ok = False
        for attempt in range(3):
            emit({"type": "trace", "role": "agent",
                  "text": f"pick_place({topping}, goal=({goal[0]:.3f},{goal[1]:.3f}))"
                          + (f"  [retry {attempt}]" if attempt else "")})
            res = runner.place(topping, goal, release_tilt=20.0, seed=attempt * 7 + 1,
                               on_frame=lambda f: emit({"type": "frame", "data": _jpeg_b64(f)}))
            j = runner.judge_added(res["before_crop_b64"], res["after_crop_b64"], topping, desc)
            emit({"type": "verdict", "topping": topping, "ok": bool(j["verdict"]),
                  "text": f"check_in_bowl({topping}) → {'YES' if j['verdict'] else 'NO'}"})
            if j["verdict"]:
                emit({"type": "trace", "role": "agent", "text": f"{topping} を確認。次へ。"})
                placed_ok = True
                break
            emit({"type": "trace", "role": "agent", "text": f"{topping} が丼に無い。やり直します。"})
        if not placed_ok:
            emit({"type": "trace", "role": "agent", "text": f"{topping} は retry 上限。次へ進みます。"})
    emit({"type": "frame", "data": _jpeg_b64(runner.render_scene())})
    emit({"type": "trace", "role": "system", "text": "盛り付け完了。"})
    emit({"type": "done"})


@app.get("/", response_class=HTMLResponse)
def index():
    return (HERE / "index.html").read_text(encoding="utf-8")


@app.websocket("/ws")
async def ws(sock: WebSocket):
    await sock.accept()
    loop = asyncio.get_running_loop()
    q: asyncio.Queue = asyncio.Queue()

    def emit(ev):  # sim スレッド → asyncio キューへ橋渡し
        loop.call_soon_threadsafe(q.put_nowait, ev)

    done = threading.Event()

    def worker():
        try:
            # 当面のUI配線検証 driver（LLM 非依存）。本命のオーケストレータは hermes-agent の
            # AIAgent（run_agent.py）を GYOZA MCP サーバ経由で駆動する形に差し替える（構築中）。
            run_assembly(emit)
        except Exception as e:  # noqa: BLE001
            emit({"type": "trace", "role": "error", "text": f"{type(e).__name__}: {e}"})
            emit({"type": "done"})
        finally:
            done.set()

    threading.Thread(target=worker, daemon=True).start()
    try:
        while not (done.is_set() and q.empty()):
            try:
                ev = await asyncio.wait_for(q.get(), timeout=0.5)
            except asyncio.TimeoutError:
                continue
            await sock.send_json(ev)
            if ev.get("type") == "done":
                break
    except WebSocketDisconnect:
        pass
