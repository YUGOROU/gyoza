"""MolmoAct2 (allenai/MolmoAct2-SO100_101) 用アダプタ。

lerobot 系統（SO101Adapter）とは別系統:
  - 状態/行動は「実機 raw スケール」(norm_stats.json の so100_so101_molmoact2 タグ):
    pan/wrist_roll ≈ 0 中心の度、lift/elbow ≈ 0-360 系の絶対度、gripper = 0-100 正規化
  - model.predict_action() が action chunk (action_horizon=30, 6) を返す（absolute joint pose）

sim との座標対応（zero-shot 近似・実機較正なしの前提を明示）:
  molmo = sim_deg + offset,  offset := state_q50 - HOME_SIM_DEG
  （sim のホームポーズを訓練分布の中央値にアンカー。gripper のみ 0-100 に直クリップ。
    関節回転方向の符号一致は menagerie MJCF と実機較正の一致を仮定 — ここが誤差源になり得る）
"""

from __future__ import annotations

import json

import numpy as np

REPO_ID = "allenai/MolmoAct2-SO100_101"
NORM_TAG = "so100_so101_molmoact2"

# PickPlaceEnv.HOME_QPOS の度数版（sim アダプタ境界のホーム状態）
HOME_SIM_DEG = np.array([0.0, -90.0, 71.6, 57.3, -90.0, 28.6])


def _load_q50() -> np.ndarray:
    from huggingface_hub import hf_hub_download

    p = hf_hub_download(REPO_ID, "norm_stats.json")
    tag = json.load(open(p))["metadata_by_tag"][NORM_TAG]
    return np.array(tag["state_stats"]["q50"], dtype=np.float64)


class MolmoActAdapter:
    """sim obs（frames + state6 sim度）→ MolmoAct2 → sim度 action chunk (T,6)。"""

    def __init__(self, device: str = "cuda", num_steps: int = 10, verbose: bool = True):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.torch = torch
        self.device = device
        self.num_steps = num_steps
        self.processor = AutoProcessor.from_pretrained(REPO_ID, trust_remote_code=True)
        self.model = AutoModelForImageTextToText.from_pretrained(
            REPO_ID, trust_remote_code=True, dtype=torch.bfloat16,
        ).to(device).eval()

        q50 = _load_q50()
        # 関節5軸はアフィン（ホーム→q50 アンカー）。gripper は 0-100 直マップ
        self.offset = q50 - HOME_SIM_DEG
        self.offset[5] = 0.0
        if verbose:
            print(f"[molmoact] state q50 = {q50.round(1).tolist()}")
            print(f"[molmoact] sim→molmo offset = {self.offset.round(1).tolist()}")

    # ---- 座標変換 ----
    def sim_to_molmo(self, state_sim_deg) -> np.ndarray:
        s = np.asarray(state_sim_deg, dtype=np.float64)[:6] + self.offset
        s[5] = float(np.clip(s[5], 0.0, 100.0))
        return s.astype(np.float32)

    def molmo_to_sim(self, actions_molmo: np.ndarray) -> np.ndarray:
        a = np.asarray(actions_molmo, dtype=np.float64)
        a = a.reshape(-1, a.shape[-1])[:, :6] - self.offset
        a[:, 5] = np.clip(a[:, 5] + 0.0, -10.0, 100.0)  # sim gripper 可動域へ
        return a

    # ---- 推論 ----
    def predict_chunk(self, frames: dict, state_sim_deg, task: str) -> np.ndarray:
        """frames: {'overhead','side'} HWC uint8 RGB → sim度 action chunk (T,6)。"""
        from PIL import Image

        torch = self.torch
        images = [Image.fromarray(frames["overhead"]), Image.fromarray(frames["side"])]
        state = self.sim_to_molmo(state_sim_deg)
        with torch.inference_mode(), torch.autocast(self.device, dtype=torch.bfloat16):
            out = self.model.predict_action(
                processor=self.processor,
                images=images,
                task=task,
                state=state,
                norm_tag=NORM_TAG,
                inference_action_mode="continuous",
                enable_depth_reasoning=False,
                num_steps=self.num_steps,
                normalize_language=True,
                enable_cuda_graph=False,  # MPS/汎用互換（MEMORY-ROBOTICS の知見）
            )
        actions = out.actions
        if hasattr(actions, "detach"):  # CUDA テンソルで返る
            actions = actions.detach().float().cpu().numpy()
        return self.molmo_to_sim(np.asarray(actions, dtype=np.float64))
