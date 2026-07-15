"""GYOZA 盛り付けデモの sim バックエンド（ツール提供体）。

エージェント（hermes-agent）が MCP/関数ツールとして呼ぶ実行体:
  - place(topping, goal_xy)      : RL+lastinch+傾け開放で丼へ配置（GT成功なら持ち越しに commit）
  - judge_added(topping)         : 丼クロップ before/after 差分を Kimi-K2.6 で二値判定（postcondition）
  - render_scene()               : 現在の積み上げ丼を俯瞰レンダ

積み上げ（A ジオラマ + ライブ orchestrator ループ）: 各配置は「現在の事前配置+持ち越し」を
静物とした env を新規構築し、把持対象トッピングだけを RL に掴ませる。GT 成功＝丼内着地なら
その実着地点を静物として持ち越す（次配置の env に含める）。retry・逐次はエージェントが判断。

実行体は当面スクリプト expert（place_skill）。datagen 完了後に学習 goal-ACT へ差し替え予定。
"""
from __future__ import annotations

import base64
import io
import os

import numpy as np


class SimRunner:
    def __init__(self, model_path: str, bowl_center=(0.204, -0.098), soup_top: float = 0.030,
                 pre_placed=None, vlm_model: str = "moonshotai/Kimi-K2.6",
                 hf_token: str | None = None):
        os.environ.setdefault("GYOZA_BOWL", f"{bowl_center[0]},{bowl_center[1]}")
        from stable_baselines3 import PPO
        self.model = PPO.load(model_path, device="cpu")
        self.bowl_center = tuple(bowl_center)
        self.soup_top = soup_top
        self.pre_placed = list(pre_placed or [])   # [(name,(x,y),yaw)] チャーシュー/メンマ
        self.placed = []                           # ライブ配置の持ち越し
        self.vlm_model = vlm_model
        self.hf_token = hf_token or os.environ.get("HF_TOKEN")
        self._bbox = None

    # ---- 内部 ----
    def _build_env(self, tomato: str, seed: int):
        from gyoza.envs.rl_pick_place import RLPickPlaceEnv
        return RLPickPlaceEnv(seed=seed, render_obs=True, tomato=tomato,
                              statics=self.pre_placed + self.placed, statics_soup_z=self.soup_top,
                              width=960, height=720)

    def _bowl_crop(self, env, frame):
        from gyoza.envs.place_skill import project_bowl_bbox
        if self._bbox is None:
            self._bbox = project_bowl_bbox(env.env, self.bowl_center, r=0.075)
        x0, y0, x1, y1 = self._bbox
        return frame[y0:y1, x0:x1]

    # ---- ツール ----
    def render_scene(self) -> np.ndarray:
        """現在の積み上げ丼（事前配置+持ち越し）を俯瞰レンダ。把持対象は板の外へ退避。"""
        env = self._build_env("sphere", seed=0)
        env.reset()
        tq = env.env._tomato_qpos
        env.env.data.qpos[tq:tq + 3] = [0.0, 0.5, -0.2]  # 赤球を画面外へ
        import mujoco
        mujoco.mj_forward(env.env.model, env.env.data)
        f = env.env.render("overhead")
        env.close()
        return f

    def place(self, topping: str, goal_xy, release_tilt: float = 20.0, seed: int = 0,
              on_frame=None, commit: bool = True) -> dict:
        """1 スキル実行。goal_xy は卓面座標（絶対）。

        commit=True: GT 成功なら即持ち越しに commit（スクリプト driver 用）。
        commit=False: commit しない（エージェント駆動用 — VLM 判定後に commit_landing で
        明示コミットし、VLM 偽陰性→retry での二重配置を避ける）。
        返り値: dict(success_gt, landing_xy, yaw, before_crop_b64, after_crop_b64, n_frames)
        on_frame(frame) が与えられれば実行中の各俯瞰フレームを渡す（ライブUI用）。
        """
        from gyoza.envs.place_skill import place_topping
        env = self._build_env(topping, seed)
        env.reset()
        frames = []

        def cap(f):
            frames.append(f)
            if on_frame is not None:
                on_frame(f)

        # goal_xy(絶対) を place_topping の goal_off(anchor 相対) に変換するため、
        # anchor は place_topping 内で実測される。ここでは絶対 goal を渡せるよう
        # goal_off = goal_xy - 予測 anchor... ではなく、place_topping を絶対ゴール対応にする。
        before = self._bowl_crop(env, env.env.render("overhead"))
        res = place_topping(env, self.model, topping, goal_xy=self._clamp_goal(goal_xy),
                            release_tilt=release_tilt, soup_top=self.soup_top,
                            capture=cap)
        after = self._bowl_crop(env, frames[-1] if frames else env.env.render("overhead"))
        success = bool(env.env.check_success())
        landing = res["landing_xy"]
        yaw = float(np.random.default_rng(seed).uniform(0, 2 * np.pi))
        if success and commit:            # 物理的に丼内 → 持ち越しに即 commit
            self.placed.append((topping, (landing[0], landing[1]), yaw))
        env.close()
        return dict(success_gt=success, landing_xy=landing, yaw=yaw,
                    before_crop_b64=_png_b64(before), after_crop_b64=_png_b64(after),
                    n_frames=len(frames))

    def commit_landing(self, topping: str, landing_xy, yaw: float) -> None:
        """エージェントが VLM 判定で「配置成功」と決めた個体を持ち越しに commit。"""
        self.placed.append((topping, (float(landing_xy[0]), float(landing_xy[1])), float(yaw)))

    # 到達可能域（RL ホバー到達点 (0.204,-0.118)+(0,0.022) 周辺 半径~1.8cm）。
    # エージェントの goal は丼中心近傍のこの円盤にクランプする（丼一面には広げられない物理制約）。
    REACH_CENTER = (0.204, -0.096)
    REACH_R = 0.018

    def _clamp_goal(self, goal_xy):
        g = np.asarray(goal_xy, dtype=float)
        c = np.asarray(self.REACH_CENTER)
        v = g - c
        n = float(np.linalg.norm(v))
        return tuple(c + v / n * self.REACH_R if n > self.REACH_R else g)

    def judge_added(self, before_crop_b64: str, after_crop_b64: str, topping: str,
                    desc: str = "") -> dict:
        """postcondition: 丼に新しく {topping} が入ったか（before/after 差分・Kimi-K2.6）。"""
        from huggingface_hub import InferenceClient
        client = InferenceClient(api_key=self.hf_token, provider="auto")
        prompt = (f"Two close-up top-down photos of the SAME ramen bowl. FIRST=before, "
                  f"SECOND=after an attempt to add a {topping}"
                  f"{f' ({desc})' if desc else ''}. Did a NEW {topping} successfully land "
                  f"inside the bowl between the two photos? Answer only yes or no.")
        msg = [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{before_crop_b64}"}},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{after_crop_b64}"}},
        ]}]
        r = client.chat.completions.create(model=self.vlm_model, messages=msg, max_tokens=512)
        text = (r.choices[0].message.content or "").strip()
        verdict = text.lower().lstrip().startswith("y")
        return dict(verdict=verdict, raw=text)


def _png_b64(rgb: np.ndarray) -> str:
    import imageio
    buf = io.BytesIO()
    imageio.imwrite(buf, rgb, format="png")
    return base64.b64encode(buf.getvalue()).decode()
