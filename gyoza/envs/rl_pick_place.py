"""RL エキスパート用 gymnasium ラッパー（状態ベース・ゴール条件付き・密報酬 v2）。

役割（RL スパイク → 教師データ工場）:
  特権状態で PPO エキスパートを学習し、収束後に俯瞰カメラ描画付きロールアウト →
  成功エピソードを goal-conditioned ACT（数値ゴール入力付き改造 ACT）への蒸留データにする。

v1 の失敗（2026-07-05, 3M steps, success 0%）:
  gripperframe 基準の reach 報酬のみでは「開いたまま押さえる」局所解に落ちた。
  → v2: 把持点 = ジョー先端中点、両ジョー接触ボーナス、接触時のみリフト進捗報酬。
v2 の失敗（2026-07-05, 3M steps, success ~0-5%）:
  把持・リフトは完全習得したが皿から ~15cm で停止保持する局所解。
  (a) tanh(10 d_place) が d>0.2 でほぼ飽和し皿方向の勾配が消失、
  (b) 成功即終了のため「+10 で打ち切り」<「~4/step × 残ステップ保持」で成功が損。
  → v3: place カーネルを tanh(4d) に緩和、成功で終了せず毎ステップ +10
    （皿に置いて居座る > 掴んで保持、の順序を保証）。終了は truncation のみ。

観測 (28,):
  [qpos6, qvel6, jaw_mid3, tomato3, tomato-jaw_mid3, plate_xy2, tomato-plate_xy2,
   contact_fixed1, contact_moving1, grasped1]
行動 (6,): [-1,1] → 関節目標増分（±MAX_DELTA rad / step @30Hz）

報酬（REWARD_VARIANT env / variant 引数で重み切替）:
  r = 1 - tanh(10 d_reach)
    + w_contact * (両ジョー接触)
    + w_lift * lift_progress            （両ジョー接触時のみ）
    + w_place * (1 - tanh(4 d_place))   （grasped 時のみ）
    + 10 * success（毎ステップ・終了しない）- 0.001|a|^2
  grasped := 両ジョー接触 かつ tomato z > GRASP_Z
"""

from __future__ import annotations

import os

import gymnasium as gym
import numpy as np

from gyoza.envs.pick_place import HOME_QPOS, PickPlaceEnv

MAX_DELTA = 0.05      # rad / step（30Hz で ~86°/s 上限）
EPISODE_STEPS = 300   # 10 s
GRASP_Z = 0.045       # トマト半径0.016 + まな板0.015 + 持ち上げ余裕
TOMATO_REST_Z = 0.031

VARIANTS = {
    # ベース: 接触・リフト・プレイスの標準重み
    "v2a": dict(w_contact=1.0, w_lift=2.0, w_place=2.0),
    # 接触を強く誘導（把持が出ない場合）
    "v2b": dict(w_contact=2.0, w_lift=2.0, w_place=2.0),
    # リフト・プレイス重視（接触は出るが運ばない場合）
    "v2c": dict(w_contact=1.0, w_lift=4.0, w_place=4.0),
    # v4: 皿ゾーン上空（d_place < release_zone）では接触・リフト報酬をゼロにし、
    #     「放しても損しない」構造にする（皿上で握り続ける局所解の対策）
    "v4a": dict(w_contact=1.0, w_lift=2.0, w_place=2.0, release_zone=0.06),
    # v5: 把持系報酬を tanh(grip_decay·d_place) で滑らかに減衰。
    #     grip=3·tanh(2d) + place=2·(1-tanh(4d)) の合計が距離によらずほぼフラット(~1.9/step)
    #     になり、ホバー農場がどこにも存在しない。成功 +10/step だけが支配的。
    #     （v3 の失敗 = 皿縁 7.4cm ホバー局所解への対策。FK 検証で皿中心は到達可能と確認済）
    "v5a": dict(w_contact=1.0, w_lift=2.0, w_place=2.0, grip_decay=2.0),
}


class RLPickPlaceEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, seed: int | None = None, variant: str | None = None,
                 render_obs: bool = False, tomato: str | None = None,
                 statics: list | None = None, statics_soup_z: float = 0.030,
                 width: int = 640, height: int = 480):
        super().__init__()
        self.w = VARIANTS[variant or os.environ.get("REWARD_VARIANT", "v2a")]
        self.env = PickPlaceEnv(width=width, height=height, seed=seed, render_obs=render_obs,
                                tomato=tomato, statics=statics, statics_soup_z=statics_soup_z)
        joints = ("shoulder_pan", "shoulder_lift", "elbow_flex",
                  "wrist_flex", "wrist_roll", "gripper")
        self._ctrl_lo = np.array([self.env.model.actuator(f"follower_{j}").ctrlrange[0] for j in joints])
        self._ctrl_hi = np.array([self.env.model.actuator(f"follower_{j}").ctrlrange[1] for j in joints])
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(28,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(6,), dtype=np.float32)
        self._target = HOME_QPOS.copy()
        self._t = 0

    # ---- gym API ----
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.env.rng = np.random.default_rng(seed)
        self.env.reset()
        self._target = HOME_QPOS.copy()
        self._t = 0
        return self._obs(), {}

    def step(self, action):
        self._target = np.clip(
            self._target + np.asarray(action, dtype=np.float64) * MAX_DELTA,
            self._ctrl_lo, self._ctrl_hi)
        self.env.step(np.rad2deg(self._target))  # PickPlaceEnv は度契約
        self._t += 1

        obs = self._obs()
        d_reach = float(np.linalg.norm(obs[18:21]))    # tomato - jaw_mid
        d_place = float(np.linalg.norm(obs[23:25]))    # tomato-plate xy
        both = bool(obs[25] > 0.5 and obs[26] > 0.5)
        grasped = bool(obs[27] > 0.5)
        tomato_z = float(obs[17])
        success = self.env.check_success()

        lift = float(np.clip((tomato_z - TOMATO_REST_Z) / 0.05, 0.0, 1.0))
        # v4: 皿ゾーン上空では把持系報酬を消し、放す行為を無償化する
        # v5: 皿に近づくほど把持系報酬を滑らかに減衰（ホバー農場の排除）
        if "grip_decay" in self.w:
            grip_gain = float(np.tanh(self.w["grip_decay"] * d_place))
        else:
            grip_gain = 0.0 if d_place < self.w.get("release_zone", 0.0) else 1.0
        reward = (1.0 - np.tanh(10.0 * d_reach)) \
            + self.w["w_contact"] * both * grip_gain \
            + self.w["w_lift"] * lift * both * grip_gain \
            + self.w["w_place"] * (1.0 - np.tanh(4.0 * d_place)) * grasped \
            - 0.001 * float(np.square(action).sum())
        if success:
            reward += 10.0
        info = {"is_success": success, "grasped": grasped, "both_contacts": both,
                "d_reach": d_reach, "d_place": d_place, "lift": lift}
        return obs, float(reward), False, self._t >= EPISODE_STEPS, info

    # ---- 内部 ----
    def _obs(self) -> np.ndarray:
        e = self.env
        jaw_mid, tomato, plate = e.jaw_mid_pos(), e.tomato_pos(), e.plate_pos()
        cf, cm = e.jaw_contacts()
        grasped = float(cf and cm and tomato[2] > GRASP_Z)
        return np.concatenate([
            e.qpos_rad(),              # 0:6
            e.qvel_rad(),              # 6:12
            jaw_mid,                   # 12:15
            tomato,                    # 15:18
            tomato - jaw_mid,          # 18:21
            plate[:2],                 # 21:23
            tomato[:2] - plate[:2],    # 23:25
            [float(cf), float(cm)],    # 25:27
            [grasped],                 # 27
        ]).astype(np.float32)

    def close(self):
        self.env.close()
