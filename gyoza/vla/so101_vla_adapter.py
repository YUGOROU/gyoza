"""SO-101 ゼロショット VLA アダプタ（純 Python・Modal 非依存）。

目的:
  cross-embodiment ベース（smolvla / pi05 / xvla / ...）の obs 形式・正規化・出力次元の差を
  吸収し、**実機 SO-101 の (overhead/side 2フレーム + 6次元 state) → 6次元 action** という
  統一インターフェースで扱う。`modal_zeroshot_smoke.py` と本番 PolicyServer の双方から import する。

設計（HANDOFF「アダプタ I/O 契約」準拠）:
  - state 入力: SO-101 実 6次元（度・slot 順 shoulder_pan/shoulder_lift/elbow_flex/
    wrist_flex/wrist_roll/gripper）をモデル state 次元の **先頭6スロット** へ、残りは 0 pad。
  - action 出力: postprocessor 後は absolute 関節目標（pi05 は relative 構成だが
    absolute_actions_processor で復元＝要 repo pipeline）。**先頭6スロット** を SO-101 指令とする。
  - camera: 実機2枚（overhead 主 + side）→ モデルの image キー（昇順）へ overhead→第1 / side→第2 /
    第3枠は overhead 複製（暫定）。
  - xvla: domain_id（soft prompt）を override（既定 0=Bridge ＝ 単腕卓上 pick-place に最近接）。

注意:
  - 契約はハードコードせず policy.config から動的抽出（キー名・次元・画像解像度のズレに強い）。
  - MolmoAct2 は raw 0–360 系で本系統と別（LeRobot 統合側で処理）。本アダプタ対象外。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

SO101_JOINTS = [
    "shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper",
]
SO101_DIM = 6

_STAT_KEYS = ("mean", "std", "min", "max", "q01", "q99")
_NEUTRAL = {"mean": 0.0, "std": 1.0, "min": -1.0, "max": 1.0, "q01": -1.0, "q99": 1.0}


# ---- stats ----------------------------------------------------------------

def load_so101_stats(stats_repo: str = "YUGOROU/act_grasp_almond"):
    """本番 SO-101 stats（meta/stats.json, 度空間6次元）の state/action を取得。"""
    import json
    from huggingface_hub import hf_hub_download

    p = hf_hub_download(stats_repo, "meta/stats.json", repo_type="dataset")
    raw = json.load(open(p))
    real_state = {k: raw["observation.state"][k] for k in _STAT_KEYS}
    real_action = {k: raw["action"][k] for k in _STAT_KEYS}
    return real_state, real_action


def _slot_mapped(dim: int, real6: dict) -> dict:
    """実 SO-101 6次元を先頭6スロットへ、残りは中立 pad（pad 出力は使わない）。"""
    out = {}
    for k in _STAT_KEYS:
        t = torch.full((dim,), float(_NEUTRAL[k]))
        t[:SO101_DIM] = torch.tensor(real6[k][:SO101_DIM], dtype=torch.float32)
        out[k] = t
    return out


def _synthetic(dim: int) -> dict:
    return {k: torch.full((dim,), float(_NEUTRAL[k])) for k in _STAT_KEYS}


# ---- adapter --------------------------------------------------------------

class SO101Adapter:
    """ロード済み policy をラップし、SO-101 obs ⇄ action を仲介する。"""

    def __init__(self, policy, pre, post, contract: dict, device: str):
        self.policy = policy
        self.pre = pre
        self.post = post
        self.c = contract  # image_keys / state_key / action_key / state_dim / action_dim / img_shapes
        self.device = device

    # --- 構築 ---
    @classmethod
    def from_policy(
        cls,
        policy,
        repo: str,
        device: str,
        *,
        stats_repo: str | None = "YUGOROU/act_grasp_almond",
        xvla_domain_id: int = 0,
        verbose: bool = True,
    ) -> "SO101Adapter":
        contract = cls._extract_contract(policy.config)
        if verbose:
            print(f"[adapter] contract: image_keys={contract['image_keys']} "
                  f"state_dim={contract['state_dim']} action_dim={contract['action_dim']}")

        pre, post = cls.build_processors(
            policy, repo, device, contract,
            stats_repo=stats_repo, xvla_domain_id=xvla_domain_id, verbose=verbose,
        )
        return cls(policy, pre, post, contract, device)

    @classmethod
    def build_processors(cls, policy, repo, device, contract=None, *,
                         stats_repo="YUGOROU/act_grasp_almond", xvla_domain_id=0,
                         rename_map=None, n_empty=1, verbose=True):
        """policy を受ける薄いラッパ（既存呼び出し互換）。実体は build_processors_for_cfg。"""
        return cls.build_processors_for_cfg(
            policy.config, repo, device, stats_repo=stats_repo,
            xvla_domain_id=xvla_domain_id, rename_map=rename_map,
            n_empty=n_empty, verbose=verbose)

    @classmethod
    def build_processors_for_cfg(cls, cfg, repo, device, *,
                                 stats_repo="YUGOROU/act_grasp_almond", xvla_domain_id=0,
                                 rename_map=None, n_empty=1, extra_pre_overrides=None,
                                 verbose=True):
        """SO-101 stats を注入し pre/post processor を構築（adapter と PolicyServer が共用）。

        cfg（policy.config・live）だけで動く。やること:
          - stats を契約次元へ slot マッピングして dataset_stats override。
          - empty_cameras を n_empty へ（実機カメラ不足ぶんを空 padding。3カメラ系の共通対処）。
          - repo pipeline 優先。(1) 版スキューで例外 or (2) マルチデータセットモデル（smolvla_base）が
            override を無視（_stats_explicitly_provided=False）なら fresh build。fresh の前に
            学習済トークナイザ長を cfg へ復元（xvla の seq 超過回避）。
          - rename_map: robot obs キー → モデル obs キー（PolicyServer ハンドシェイク由来）。
          - extra_pre_overrides: 追加 preprocessor_overrides（呼び出し側固有）。
        """
        contract = cls._extract_contract(cfg)

        # 実機カメラ不足を空 padding（smolvla/xvla/pi05 とも 3 枠要求・実機 2 枚）。
        if hasattr(cfg, "empty_cameras"):
            cur = int(getattr(cfg, "empty_cameras", 0) or 0)
            if cur < n_empty:
                cfg.empty_cameras = n_empty
                if verbose:
                    print(f"[adapter] empty_cameras {cur} → {n_empty}（欠損カメラを空 padding）")

        # stats（本番 or 合成）を契約次元へ slot マッピング
        if stats_repo:
            real_state, real_action = load_so101_stats(stats_repo)
            mk_s = lambda d: _slot_mapped(d, real_state)
            mk_a = lambda d: _slot_mapped(d, real_action)
            if verbose:
                print(f"[adapter] real SO-101 stats from {stats_repo}")
        else:
            mk_s = mk_a = _synthetic
            if verbose:
                print("[adapter] synthetic stats")

        stats = {}
        if contract["state_key"]:
            stats[contract["state_key"]] = mk_s(contract["state_dim"])
        if contract["action_key"]:
            stats[contract["action_key"]] = mk_a(contract["action_dim"])

        # factory は版により lerobot.policies.factory / lerobot.policies のいずれか
        try:
            from lerobot.policies.factory import make_pre_post_processors
        except ImportError:
            from lerobot.policies import make_pre_post_processors

        pre_overrides = {"device_processor": {"device": device}}
        if rename_map:
            pre_overrides["rename_observations_processor"] = {"rename_map": rename_map}
        if getattr(cfg, "type", None) == "xvla":
            pre_overrides["xvla_add_domain_id"] = {"domain_id": xvla_domain_id}
            if verbose:
                print(f"[adapter] xvla domain_id={xvla_domain_id}")
        if extra_pre_overrides:
            pre_overrides.update(extra_pre_overrides)

        def _build(pretrained_path):
            return make_pre_post_processors(
                cfg, pretrained_path,
                dataset_stats=stats, preprocessor_overrides=pre_overrides,
            )

        repo_pre = None
        try:
            pre, post = _build(repo)
            repo_pre = pre
            if cls._stats_override_applied(post):
                if verbose:
                    print("[adapter] processors: repo pipeline (stats override 適用済)")
                return pre, post
            if verbose:
                print("[adapter] repo pipeline は stats override を無視（multi-dataset buffer）→ fresh build")
        except Exception as e:
            if verbose:
                print(f"[adapter] repo pipeline failed ({type(e).__name__}); fresh build")

        # fresh build へ切替。ただし fresh は config 値からトークナイザを再構成するため、
        # config が max_len_seq と不整合な tokenizer_max_length を持つモデル（xvla-base:
        # tokenizer_max_length=1024 だが max_len_seq=512・学習済 step は 50）では seq 超過で落ちる。
        # → repo pipeline が持つ「学習済トークナイザ長」を cfg に復元してから fresh build する。
        cls._sync_tokenizer_max_length(cfg, repo_pre, verbose=verbose)
        return _build(None)

    @classmethod
    def _sync_tokenizer_max_length(cls, cfg, repo_pre, verbose=True):
        """repo pipeline の TokenizerProcessorStep.max_length（学習済値）を cfg に反映。

        xvla-base は config.tokenizer_max_length=1024 だが学習済 pipeline は 50。fresh build は
        config 値（1024）で padding し max_len_seq=512 を超過するため、学習済値へ下げて整合させる。
        """
        if repo_pre is None or not hasattr(cfg, "tokenizer_max_length"):
            return
        trained = None
        for s in cls._iter_steps(repo_pre):
            if "oken" in type(s).__name__.lower():
                trained = getattr(s, "max_length", None)
                break
        cur = getattr(cfg, "tokenizer_max_length", None)
        if trained and cur and trained < cur:
            cfg.tokenizer_max_length = trained
            if verbose:
                print(f"[adapter] tokenizer_max_length {cur} → {trained}"
                      f"（学習済 pipeline に整合・seq 超過回避）")

    @staticmethod
    def _iter_steps(pipe):
        for attr in ("steps", "processors", "_steps", "_processors"):
            v = getattr(pipe, attr, None)
            if v and not callable(v):
                return list(v)
        try:
            return list(pipe)
        except TypeError:
            return []

    @classmethod
    def _stats_override_applied(cls, post) -> bool:
        """post の unnormalizer が我々の dataset_stats を採用したか（_stats_explicitly_provided）。"""
        for s in cls._iter_steps(post):
            if "nnormal" in type(s).__name__.lower():
                return bool(getattr(s, "_stats_explicitly_provided", False))
        return False

    @staticmethod
    def _extract_contract(cfg) -> dict:
        feats = dict(cfg.input_features)
        out_feats = dict(cfg.output_features)
        image_keys = sorted(k for k in feats if "image" in k.lower())  # 昇順＝主視点が先頭
        state_key = next((k for k in feats if k.endswith("state")), None)
        action_key = next((k for k in out_feats if "action" in k.lower()), None)
        return {
            "image_keys": image_keys,
            "state_key": state_key,
            "action_key": action_key,
            "state_dim": feats[state_key].shape[0] if state_key else None,
            "action_dim": out_feats[action_key].shape[0] if action_key else None,
            "img_shapes": {k: tuple(feats[k].shape) for k in image_keys},  # (C,H,W)
        }

    # --- obs 構築 ---
    def build_obs(self, frames: dict, state6, task: str) -> dict:
        """frames: {'overhead': HxWx3, 'side': HxWx3}（uint8/float, RGB）。state6: 長さ6。"""
        c = self.c
        # camera 割当: overhead→第1 image_key / side→第2 / 第3以降は overhead 複製
        order = ["overhead", "side"]
        obs = {}
        for i, k in enumerate(c["image_keys"]):
            src = order[i] if i < len(order) else "overhead"
            obs[k] = self._prep_image(frames[src], c["img_shapes"][k])
        if c["state_key"]:
            st = torch.zeros(1, c["state_dim"], device=self.device)
            sv = torch.as_tensor(state6, dtype=torch.float32, device=self.device)[:SO101_DIM]
            st[0, :SO101_DIM] = sv
            obs[c["state_key"]] = st
        obs["task"] = [task]
        if getattr(self, "_debug_obs", False):
            print("[adapter.build_obs] " + ", ".join(
                f"{k}={tuple(v.shape)}" for k, v in obs.items() if torch.is_tensor(v)))
        return obs

    def _prep_image(self, img, shape_chw):
        """HxWx3 or CxHxW（uint8/float）→ (1,C,H,W) float[0,1] にモデル解像度へ resize。"""
        c, h, w = shape_chw
        t = torch.as_tensor(img, dtype=torch.float32)
        if t.ndim == 3 and t.shape[-1] == 3:  # HWC → CHW
            t = t.permute(2, 0, 1)
        if t.max() > 1.5:  # uint8 想定 → [0,1]
            t = t / 255.0
        t = t.unsqueeze(0)  # (1,C,H,W)
        if t.shape[-2:] != (h, w):
            t = F.interpolate(t, size=(h, w), mode="bilinear", align_corners=False)
        return t.to(self.device)

    # --- 推論 ---
    @torch.inference_mode()
    def predict(self, frames: dict, state6, task: str = "Pick up the object and place it into the bowl."):
        """→ SO-101 6次元 absolute 関節目標（numpy, 度）。"""
        obs = self.build_obs(frames, state6, task)
        batch = self.pre(dict(obs))
        act = self.policy.select_action(batch)
        out = self.post(act)
        return self.decode(out)

    def decode(self, out):
        """モデル出力（pad 空間 (B, action_dim)）→ 先頭6スロットを SO-101 action として numpy 化。"""
        t = out if torch.is_tensor(out) else out.get(self.c["action_key"], out)
        t = t.reshape(-1)[:SO101_DIM]
        return t.detach().float().cpu().numpy()
