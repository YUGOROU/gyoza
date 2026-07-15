"""GYOZA 盛り付けツールの MCP サーバ（FastMCP）— 実 SimRunner バックエンド。

D7 の正しい構造: hermes-agent の AIAgent が MCP 経由でこれらのツールを呼び、盛り付けループ
（配置→判定→retry→次）を駆動する。手書きループは書かない。構造実証（jobs/hermes_mcp_drive_job.py・
モック）で AIAgent↔MCP↔ツール往復は確認済み。ここでモックを実 SimRunner に差し替える。

ツール（構成B 型 = エージェントはテキスト LLM、視覚判定は check_in_bowl 内で完結）:
  - pick_place(object, goal_x, goal_y): SimRunner.place で丼へ配置（commit=False）。
    実行結果テキストを返す。丼クロップ before/after は次の check_in_bowl 用に保持。
  - check_in_bowl(object): 直近 pick_place の丼クロップ before/after を Kimi-K2.6 で二値判定
    （SimRunner.judge_added）。yes かつ物理成功なら実着地点を持ち越しに commit。
  - reset_bowl(): 事前配置＋持ち越しを初期化（新しい盛り付けの開始）。

フレーム配信: place 実行中の各俯瞰フレームは、set_frame_sink() で登録した callback へ流す
（Space の websocket キューへ橋渡し。stdio 検証ジョブでは未登録＝カウントのみ）。

起動:
  - stdio（検証ジョブ・hermes config の command/args 登録）: `python gyoza_mcp.py`
  - http（Space に同一プロセスで mount）: `build_mcp(runner)` で FastMCP を得て ASGI マウント
"""
from __future__ import annotations

import os

from fastmcp import FastMCP

# フレーム sink（Space が websocket キューへ橋渡しする callback を登録）。既定 None=カウントのみ。
_frame_sink = None


def set_frame_sink(fn) -> None:
    global _frame_sink
    _frame_sink = fn


def build_mcp(runner) -> FastMCP:
    """SimRunner を束ねた FastMCP サーバを構築（stdio/http 共用）。"""
    mcp = FastMCP("gyoza")
    # 直近 pick_place の状態（check_in_bowl / commit 用）
    st: dict = {"last": None}

    def _on_frame(f):
        if _frame_sink is not None:
            _frame_sink(f)

    @mcp.tool
    def pick_place(object: str, goal_x: float, goal_y: float) -> str:
        """Grasp a ramen topping and place it at a goal point in the bowl.

        The reachable zone is a small disk (~1.8cm) around the bowl center
        (0.204, -0.096) in table coordinates (meters); goals are clamped to it.
        Returns a short text describing the outcome. Always follow with check_in_bowl.
        """
        # retry ごとに把持初期条件を変える（seed）。同一トッピングの試行回数で seed を進める。
        n = st["last"]["retries"].get(object, 0) if st["last"] else 0
        seed = n * 7 + 1
        res = runner.place(object, (goal_x, goal_y), release_tilt=20.0, seed=seed,
                           on_frame=_on_frame, commit=False)
        retries = (st["last"]["retries"] if st["last"] else {})
        retries[object] = retries.get(object, 0) + 1
        st["last"] = dict(object=object, landing=res["landing_xy"], yaw=res["yaw"],
                          gt=res["success_gt"], before=res["before_crop_b64"],
                          after=res["after_crop_b64"], retries=retries)
        return (f"pick_place({object}) executed at ({goal_x:.3f},{goal_y:.3f}); "
                f"landing=({res['landing_xy'][0]:.3f},{res['landing_xy'][1]:.3f}). "
                f"Call check_in_bowl to verify.")

    @mcp.tool
    def check_in_bowl(object: str) -> str:
        """Visually verify (vision model) whether the topping landed inside the bowl
        in the latest pick_place. Returns 'yes' or 'no'. On 'yes', the placement is
        committed so the topping stays in the bowl for subsequent placements."""
        last = st["last"]
        if last is None or last["object"] != object:
            return "no (no matching pick_place to verify)"
        j = runner.judge_added(last["before"], last["after"], object,
                               DESC.get(object, ""))
        ok = bool(j["verdict"])
        if ok and last["gt"]:
            runner.commit_landing(object, last["landing"], last["yaw"])
        return "yes" if ok else "no"

    @mcp.tool
    def reset_bowl() -> str:
        """Reset the bowl to only the pre-placed toppings (start a fresh plating)."""
        runner.placed.clear()
        st["last"] = None
        return "bowl reset to pre-placed toppings"

    return mcp


DESC = {
    "naruto": "pink-and-white spiral fish cake (narutomaki)",
    "ajitama": "halved soft-boiled egg (ajitama)",
}


def _default_runner():
    import pathlib
    import sys
    here = pathlib.Path(__file__).resolve().parent
    sys.path.insert(0, str(here.parent))   # gyoza package
    sys.path.insert(0, str(here))          # sim_runner
    from sim_runner import SimRunner
    model = os.environ.get("RL_MODEL", str(here / "best_model.zip"))
    cx, cy = (float(v) for v in os.environ.get("GYOZA_BOWL_CENTER", "0.204,-0.098").split(","))
    pre = [("chashu", (cx - 0.016, cy - 0.012), 0.3), ("menma", (cx - 0.004, cy + 0.018), -0.6)]
    return SimRunner(model, bowl_center=(cx, cy), soup_top=0.030, pre_placed=pre)


if __name__ == "__main__":
    # stdio 起動（hermes config の command/args から呼ばれる）
    build_mcp(_default_runner()).run()
