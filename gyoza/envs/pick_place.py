"""GYOZA pick_place 環境（MuJoCo・プレースホルダアセット版）。

スキル契約（gyoza-research.md D4）:
  postcondition: "object inside target_zone" — GT 判定は check_success()（幾何）。
  VLM 二値判定は俯瞰フレームに対して別レイヤで行う（本環境は正解ラベル側）。

アダプタ I/O 契約（VLA-Bench so101_vla_adapter 準拠）:
  obs   : frames {'overhead','side'} HWC uint8 RGB + state 6次元（度、
          slot 順 shoulder_pan/shoulder_lift/elbow_flex/wrist_flex/wrist_roll/gripper）
  action: 6次元 absolute 関節目標（度）→ 本環境で rad へ変換し follower 腕へ指令。

単位規約: MJCF は rad、アダプタ境界は度。gripper も同様に rad↔deg 変換のみ
（実機 LeRobot の 0-100 正規化とは異なるが、ゼロショット sim 測定では度で統一）。
"""

from __future__ import annotations

import os
import pathlib

import mujoco
import numpy as np

ASSET_XML = pathlib.Path(__file__).resolve().parent.parent / "assets" / "pick_place.xml"

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

# follower 腕ホームポーズ（rad）: 作業領域を向いて軽く構えた姿勢
HOME_QPOS = np.array([0.0, -1.57, 1.25, 1.0, -1.57, 0.5])
# helper 腕は畳んで固定（本実験では不使用）
HELPER_QPOS = np.array([0.0, -1.7, 1.6, 1.4, 0.0, 0.0])

# トマト初期位置のランダム化領域（まな板上 xy。まな板 = 中心(0.09,-0.06) 半サイズ(0.19,0.14)）
TOMATO_X = (-0.06, 0.06)
TOMATO_Y = (-0.16, -0.00)
TOMATO_Z0 = 0.035  # まな板天面(0.015) + 半径 + マージン

CONTROL_HZ = 30.0

# 選別クープの果物カラーパレット（白玉/ぶどう/未熟さくらんぼ/熟さくらんぼ）
FRUIT_PALETTE = [
    (0.96, 0.95, 0.92),  # white
    (0.45, 0.20, 0.55),  # purple
    (0.15, 0.60, 0.20),  # green
    (0.85, 0.12, 0.08),  # red
]


# veg_pipeline 出力の全高 [m]（底面 z=0 整列済み。body 中心に置くための z オフセット計算用）
VEG_HEIGHTS = {
    "mini_tomato": 0.030,
    "naruto": 0.025,    # 厚切りナルト φ40mm（2026-07-08 z 押し出しで 15.7→25mm）
    "chashu": 0.022,    # 厚切りチャーシュー φ50mm（同 11.8→22mm）
    "ajitama": 0.0158,  # 味玉半切り（断面上向き）35mm
    "menma": 0.0063,    # メンマ 50mm
    # 選別クープの果物（2026-07-10、z extent = veg_pipeline 実測）
    "shiratama": 0.0294,
    # さくらんぼは本体 φ26 基準の再処理後（茎込み全高。2026-07-10）
    "cherry_stage1_green": 0.0475,
    "cherry_stage2_yellowpink": 0.0375,
    "cherry_stage3_red": 0.0365,
    "grape": 0.0193,
}

# 食材別スライド摩擦（未記載は XML の球の値 1.0 を継承）。
# 平底の薄物はリリース傾け（rollout_job RELEASE_TILT）で固定ジョーから滑落させる
# 必要があり、fric=1.0 だと落ちない。ep002 再生実測: 0.6 で滑落成功・運搬保持も両立、
# 0.4 は滑りすぎてゾーン外、1.0 は退避ですくい上げ。表面がツルツルの食材として物理的にも正当。
VEG_FRICTION = {
    "naruto": 0.6,
    "chashu": 0.6,
}


def _patch_tomato_mesh(spec: "mujoco.MjSpec", name: str = "mini_tomato") -> None:
    """把持対象の球プレースホルダを TRELLIS メッシュ（視覚 + CoACD 凸包群）へ差し替える。

    veg_pipeline.py の出力（gyoza/assets/veg/<name>/）を前提。単一シーンソース維持のため
    XML を複製せずランタイムパッチで行う（測定の既定は sphere、デモ等で mesh を opt-in）。
    body 名・geom 名は "tomato" のまま維持する（環境コードの参照を変えないため）。
    """
    veg = pathlib.Path(__file__).resolve().parent.parent / "assets" / "veg" / name
    n_hulls = len(list(veg.glob("collision_*.obj")))
    assert n_hulls > 0, f"{veg} に CoACD 出力が無い（scripts/veg_pipeline.py を先に実行）"
    z_off = -VEG_HEIGHTS[name] / 2  # 底面 z=0 のメッシュを body 中心へ

    spec.add_texture(name=f"{name}_tex", type=mujoco.mjtTexture.mjTEXTURE_2D,
                     file=str(veg / "texture.png"))
    mat = spec.add_material(name=f"{name}_mat")
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = f"{name}_tex"
    mat.specular, mat.shininess = 0.1, 0.1  # 食材はマット寄り（テカリ抑制）
    spec.add_mesh(name=f"{name}_visual", file=str(veg / "visual.obj"))
    for i in range(n_hulls):
        spec.add_mesh(name=f"{name}_col{i}", file=str(veg / f"collision_{i}.obj"))

    body = spec.body("tomato")
    old = body.geoms[0]  # tomato_geom (sphere)
    friction, condim, priority, mass = old.friction.copy(), old.condim, old.priority, 0.012
    if name in VEG_FRICTION:
        friction[0] = VEG_FRICTION[name]
    spec.delete(old)
    body.add_geom(name="tomato_visual", type=mujoco.mjtGeom.mjGEOM_MESH,
                  meshname=f"{name}_visual", material=f"{name}_mat",
                  pos=[0, 0, z_off], contype=0, conaffinity=0, group=2, mass=0)
    for i in range(n_hulls):
        body.add_geom(name=f"tomato_col{i}", type=mujoco.mjtGeom.mjGEOM_MESH,
                      meshname=f"{name}_col{i}", pos=[0, 0, z_off],
                      friction=friction, condim=condim, priority=priority,
                      mass=mass / n_hulls, group=3)


def add_static_topping(spec: "mujoco.MjSpec", name: str, xy, yaw: float = 0.0,
                       soup_z: float = 0.030, idx: int = 0, group: int = 0) -> None:
    """veg アセットを静物（固定 body・視覚メッシュのみ contype=0）として丼のスープ面に置く。

    盛り付けデモの事前配置（チャーシュー/メンマ）・積み上げの持ち越し（成功済みトッピング）用。
    衝突は持たせない（トッピング同士は現実でも重なる＝視覚オーバーラップで自然。物理の暴れも回避）。
    mesh/material/texture 名は idx で一意化（同種トッピングの複数配置に対応）。
    group=4 で観測マスキング: 既定オプションのレンダラ（policy 入力）には映らず、
    geomgroup[4] を有効化した判定・動画用レンダラにのみ映る（対象指定の実装。実機の
    セグメンテーションマスクに相当）。"""
    veg = pathlib.Path(__file__).resolve().parent.parent / "assets" / "veg" / name
    h = VEG_HEIGHTS[name]
    tag = f"st{idx}_{name}"
    spec.add_texture(name=f"{tag}_tex", type=mujoco.mjtTexture.mjTEXTURE_2D,
                     file=str(veg / "texture.png"))
    mat = spec.add_material(name=f"{tag}_mat")
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = f"{tag}_tex"
    mat.specular, mat.shininess = 0.1, 0.1
    spec.add_mesh(name=f"{tag}_vis", file=str(veg / "visual.obj"))
    b = spec.worldbody.add_body(name=f"static_{tag}", pos=[xy[0], xy[1], soup_z + h / 2],
                                quat=[np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
    b.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=f"{tag}_vis", material=f"{tag}_mat",
               pos=[0, 0, -h / 2], contype=0, conaffinity=0, mass=0, group=group)


class PickPlaceEnv:
    """pick_place 環境。把持対象を皿ゾーンへ。
    tomato='sphere'(既定・測定用) | 'mesh'(=mini_tomato) | VEG_HEIGHTS の任意アセット名。
    statics: [(name, (x,y), yaw), ...] 事前配置/持ち越しの静物トッピング（盛り付けデモ用）。"""

    def __init__(self, width: int = 640, height: int = 480, seed: int | None = None,
                 tomato: str | None = None, render_obs: bool = True,
                 statics: list | None = None, statics_soup_z: float = 0.030,
                 statics_group: int = 0):
        tomato = tomato or os.environ.get("GYOZA_TOMATO", "sphere")
        self._tomato_kind = tomato
        spec = mujoco.MjSpec.from_file(str(ASSET_XML))
        if tomato == "mesh":
            _patch_tomato_mesh(spec, "mini_tomato")
        elif tomato != "sphere":
            _patch_tomato_mesh(spec, tomato)
        # 丼シーン（デモ専用・opt-in）: 実験用の平皿 body をランタイムで丼へ差し替える。
        # GYOZA_BOWL="x,y" で丼中心を指定（"1"/"default" なら patch_bowl の既定中心）。
        # スープ面 z を現皿天面(0.012)に揃えるため expert/ACT は無改修で移植できる。
        bowl_env = os.environ.get("GYOZA_BOWL", "")
        if bowl_env and bowl_env not in ("0", "sphere"):
            from gyoza.envs.bowl import patch_bowl
            if bowl_env in ("1", "default"):
                patch_bowl(spec)
            else:
                cx, cy = (float(v) for v in bowl_env.split(","))
                patch_bowl(spec, center_xy=(cx, cy))
        # クープグラス（選別つきクープ盛り付け・opt-in）: GYOZA_COUPE="x,y" or "1"/"default"。
        # GYOZA_BOWL と排他（両方指定時は COUPE 優先）。底面 z=0.012=平皿天面で ACT 無改修移植。
        coupe_env = os.environ.get("GYOZA_COUPE", "")
        coupe_floor_z = None
        if coupe_env and coupe_env != "0":
            from gyoza.envs.bowl import patch_coupe
            if coupe_env in ("1", "default"):
                coupe_floor_z = patch_coupe(spec)
            else:
                cx, cy = (float(v) for v in coupe_env.split(","))
                coupe_floor_z = patch_coupe(spec, center_xy=(cx, cy))
        # ライブ球の色/半径パッチ（選別デモの色球・半径スパイク用。sphere のみ有効）:
        #   GYOZA_SPHERE_RGBA="r,g,b[,a]"（tomato_mat を書き換え）
        #   GYOZA_SPHERE_R="0.013"（球半径 [m]）
        if tomato == "sphere":
            rgba_env = os.environ.get("GYOZA_SPHERE_RGBA", "")
            if rgba_env and rgba_env != "rand":
                v = [float(x) for x in rgba_env.split(",")]
                spec.material("tomato_mat").rgba = v + [1.0] * (4 - len(v))
            r_env = os.environ.get("GYOZA_SPHERE_R", "")
            if r_env:
                spec.body("tomato").geoms[0].size[0] = float(r_env)
        # 距離物（静物色球・視覚のみ contype=0）: 選別シナリオの「カウンターの果物」。
        #   GYOZA_DISTRACTORS="x,y,r,R,G,B;..."（固定配置。まな板天面 z=0.015+r に置く）
        #   GYOZA_DISTRACT_RAND="N"（リセット毎に N 個を位置・色ランダム化 — datagen/eval 用）
        for i, seg in enumerate(filter(None, os.environ.get("GYOZA_DISTRACTORS", "").split(";"))):
            x, y, r, cr, cg, cb = (float(v) for v in seg.split(","))
            spec.worldbody.add_geom(
                name=f"distractor_{i}", type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[r, 0, 0], pos=[x, y, 0.015 + r], rgba=[cr, cg, cb, 1.0],
                contype=0, conaffinity=0, mass=0)
        self._n_rand_distract = int(os.environ.get("GYOZA_DISTRACT_RAND", "0"))
        for i in range(self._n_rand_distract):  # プレースホルダ生成（位置・色は reset で決める）
            spec.worldbody.add_geom(
                name=f"rdistract_{i}", type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[0.013, 0, 0], pos=[0.5, 0.5, -0.1], rgba=[1, 1, 1, 1],
                contype=0, conaffinity=0, mass=0)
        # 実メッシュ距離物（GYOZA_DISTRACT_MESH="shiratama,grape,..."）: 各スロットに
        # veg アセットの静物 body を生成し、位置・向きは reset 毎にランダム化。
        # 本番の選別シーン（果物が並ぶまな板）と視覚一致させる datagen 用。
        self._mesh_distract = [s for s in os.environ.get("GYOZA_DISTRACT_MESH", "").split(",") if s]
        for i, nm in enumerate(self._mesh_distract):
            add_static_topping(spec, nm, (0.5, 0.5 + 0.1 * i), 0.0, soup_z=-0.5, idx=100 + i)
        # ライブ球の色ランダム化: GYOZA_SPHERE_RGBA="rand"（reset 毎にパレットから抽選）
        self._rand_sphere_color = os.environ.get("GYOZA_SPHERE_RGBA", "") == "rand"
        # 静物トッピング（盛り付けデモの事前配置・持ち越し）。
        # 要素は (name, (x,y), yaw) または (name, (x,y), yaw, z)（z 省略時は statics_soup_z。
        # 選別クープではまな板上の果物 z=0.015 とクープ内持ち越し z=0.058 が混在するため）
        for i, s in enumerate(statics or []):
            nm, xy, yaw = s[0], s[1], s[2]
            z = s[3] if len(s) > 3 else statics_soup_z
            add_static_topping(spec, nm, xy, yaw, soup_z=z, idx=i, group=statics_group)
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        self.rng = np.random.default_rng(seed)
        # render_obs=False: RL 等の状態オンリー用途（Renderer 生成も描画もしない）
        self.render_obs = render_obs
        self.renderer = mujoco.Renderer(self.model, height, width) if render_obs else None
        self.n_substeps = max(1, round(1.0 / CONTROL_HZ / self.model.opt.timestep))

        self._fol_jnt = [self.model.joint(f"follower_{j}").qposadr[0] for j in JOINTS]
        self._fol_act = [self.model.actuator(f"follower_{j}").id for j in JOINTS]
        self._helper_jnt = [self.model.joint(f"helper_{j}").qposadr[0] for j in JOINTS]
        self._helper_act = [self.model.actuator(f"helper_{j}").id for j in JOINTS]
        self._tomato_qpos = self.model.joint("tomato_free").qposadr[0]
        self._tomato_body = self.model.body("tomato").id
        self._plate_site = self.model.site("plate_center").id
        self._tcp_site = self.model.site("follower_gripperframe").id  # TCP（グリッパー先端参照点）
        # 把持系 geom id 集合（jaw_mid_pos / jaw_contacts 用）
        def _geom_ids(pred):
            return {i for i in range(self.model.ngeom)
                    if (n := self.model.geom(i).name) and pred(n)}
        self._fixed_tip_ids = sorted(_geom_ids(lambda n: n.startswith("follower_fixed_jaw_sph_tip")))
        self._moving_tip_ids = sorted(_geom_ids(lambda n: n.startswith("follower_moving_jaw_sph_tip")))
        self._fixed_jaw_ids = _geom_ids(lambda n: n.startswith("follower_fixed_jaw"))
        self._moving_jaw_ids = _geom_ids(lambda n: n.startswith("follower_moving_jaw"))
        tomato_body_id = self._tomato_body
        self._tomato_geom_ids = {i for i in range(self.model.ngeom)
                                 if self.model.geom_bodyid[i] == tomato_body_id}
        self._ctrl_lo = self.model.actuator_ctrlrange[:, 0].copy()
        self._ctrl_hi = self.model.actuator_ctrlrange[:, 1].copy()

        # 皿ゾーン: 円盤中心から半径・高さ許容
        self.plate_radius = 0.055
        self.zone_z = (0.005, 0.06)
        if coupe_floor_z is not None:  # クープは内底が高い → ゾーン z を内底基準に
            self.zone_z = (coupe_floor_z - 0.005, coupe_floor_z + 0.05)

    # ---- ライフサイクル ----
    def reset(self) -> dict:
        mujoco.mj_resetData(self.model, self.data)
        for adr, q in zip(self._fol_jnt, HOME_QPOS):
            self.data.qpos[adr] = q
        for adr, q in zip(self._helper_jnt, HELPER_QPOS):
            self.data.qpos[adr] = q
        # トマト初期位置ランダム化（実験計画 §4: 物体初期姿勢ランダム化）
        x = self.rng.uniform(*TOMATO_X)
        y = self.rng.uniform(*TOMATO_Y)
        self.data.qpos[self._tomato_qpos : self._tomato_qpos + 7] = [x, y, TOMATO_Z0, 1, 0, 0, 0]
        # ライブ球の色ランダム化（GYOZA_SPHERE_RGBA="rand"・視覚頑健化 datagen 用）
        live_color = None
        if self._rand_sphere_color:
            live_color = FRUIT_PALETTE[self.rng.integers(len(FRUIT_PALETTE))]
            mat_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_MATERIAL, "tomato_mat")
            self.model.mat_rgba[mat_id, :3] = live_color
        # 距離物ランダム化（GYOZA_DISTRACT_RAND=N・視覚のみ）: まな板上・ライブ球と
        # ≥5cm / 相互 ≥3.5cm 離す。色はパレットからライブ球色を除いて抽選（把持対象の
        # 視覚的一意性を保つ — 同色があると ACT がどれを掴むか原理的に曖昧になる）
        if self._n_rand_distract:
            palette = [c for c in FRUIT_PALETTE if c != live_color] if live_color else FRUIT_PALETTE
            placed_xy = [(x, y)]
            for i in range(self._n_rand_distract):
                for _ in range(200):
                    dx = self.rng.uniform(-0.08, 0.16)
                    dy = self.rng.uniform(-0.17, 0.05)
                    if (np.hypot(dx - x, dy - y) > 0.05
                            and all(np.hypot(dx - px, dy - py) > 0.035
                                    for px, py in placed_xy[1:])):
                        break
                placed_xy.append((dx, dy))
                r = float(self.rng.choice([0.013, 0.016]))
                gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"rdistract_{i}")
                self.model.geom_size[gid, 0] = r
                self.model.geom_pos[gid] = [dx, dy, 0.015 + r]
                self.model.geom_rgba[gid, :3] = palette[self.rng.integers(len(palette))]
        # 実メッシュ距離物の位置・向きランダム化（ライブ球 ≥5cm / 相互 ≥3.5cm はスフィア版と同一）
        if self._mesh_distract:
            placed_xy = [(x, y)]
            for i, nm in enumerate(self._mesh_distract):
                for _ in range(200):
                    dx = self.rng.uniform(-0.08, 0.16)
                    dy = self.rng.uniform(-0.17, 0.05)
                    if (np.hypot(dx - x, dy - y) > 0.05
                            and all(np.hypot(dx - px, dy - py) > 0.035
                                    for px, py in placed_xy[1:])):
                        break
                placed_xy.append((dx, dy))
                bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY,
                                        f"static_st{100 + i}_{nm}")
                self.model.body_pos[bid] = [dx, dy, 0.015 + VEG_HEIGHTS[nm] / 2]
                yaw = self.rng.uniform(0, 2 * np.pi)
                self.model.body_quat[bid] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        # ctrl をホームに合わせ、落下・整定
        self.data.ctrl[self._fol_act] = HOME_QPOS
        self.data.ctrl[self._helper_act] = HELPER_QPOS
        mujoco.mj_forward(self.model, self.data)
        for _ in range(50):
            mujoco.mj_step(self.model, self.data)
        return self.get_obs()

    def step(self, action_deg: np.ndarray) -> dict:
        """action: 6次元 absolute 関節目標（度）→ follower へ。1 制御周期（~30Hz）進める。"""
        target = np.deg2rad(np.asarray(action_deg, dtype=np.float64)[:6])
        target = np.clip(target, self._ctrl_lo[self._fol_act], self._ctrl_hi[self._fol_act])
        self.data.ctrl[self._fol_act] = target
        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)
        return self.get_obs()

    # ---- 観測 ----
    def get_obs(self) -> dict:
        obs = {"state": self.state_deg()}
        if self.render_obs:
            obs["frames"] = {"overhead": self.render("overhead"), "side": self.render("side")}
        return obs

    def state_deg(self) -> np.ndarray:
        return np.rad2deg(self.data.qpos[self._fol_jnt]).astype(np.float32)

    def render(self, camera: str = "overhead") -> np.ndarray:
        self.renderer.update_scene(self.data, camera=camera)
        return self.renderer.render()

    # ---- 事後条件（GT） ----
    def check_success(self) -> bool:
        """postcondition: tomato inside plate zone（静止条件込み）。"""
        tomato = self.data.xpos[self._tomato_body]
        plate = self.data.site_xpos[self._plate_site]
        in_xy = np.linalg.norm(tomato[:2] - plate[:2]) < self.plate_radius
        in_z = self.zone_z[0] < tomato[2] < self.zone_z[1]
        # cvel = [角速度(3), 並進速度(3)]。球（実験用・76%）は従来どおり 6D ノルムで
        # 静止判定し byte-identical に保存する。トッピング（薄い円盤等）は皿上でその場
        # 回転しても並進的に静止していれば盛り付け成功とみなす → 並進速度のみで判定。
        # naruto の「置けたのに回転で vel>0.05 → False」偽陰性（demo/datagen 両方）を解消。
        cvel = self.data.cvel[self._tomato_body]
        vel = np.linalg.norm(cvel) if self._tomato_kind == "sphere" else np.linalg.norm(cvel[3:])
        return bool(in_xy and in_z and vel < 0.05)

    def tomato_pos(self) -> np.ndarray:
        return self.data.xpos[self._tomato_body].copy()

    # ---- 特権状態（RL・自動ラベル用。実機には存在しない情報）----
    def tcp_pos(self) -> np.ndarray:
        return self.data.site_xpos[self._tcp_site].copy()

    def jaw_mid_pos(self) -> np.ndarray:
        """把持点 = 固定/可動ジョー先端球の中点（gripperframe はジョーから 2-3.5cm 手前でズレる）。"""
        f = np.mean([self.data.geom_xpos[i] for i in self._fixed_tip_ids], axis=0)
        m = np.mean([self.data.geom_xpos[i] for i in self._moving_tip_ids], axis=0)
        return (f + m) / 2

    def jaw_contacts(self) -> tuple[bool, bool]:
        """(固定ジョー接触, 可動ジョー接触) — トマトの衝突 geom との接触を走査。"""
        fixed = moving = False
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            pair = {c.geom1, c.geom2}
            if not (pair & self._tomato_geom_ids):
                continue
            if pair & self._fixed_jaw_ids:
                fixed = True
            if pair & self._moving_jaw_ids:
                moving = True
        return fixed, moving

    def plate_pos(self) -> np.ndarray:
        return self.data.site_xpos[self._plate_site].copy()

    def qpos_rad(self) -> np.ndarray:
        return self.data.qpos[self._fol_jnt].copy()

    def qvel_rad(self) -> np.ndarray:
        return np.array([self.data.qvel[self.model.joint(f"follower_{j}").dofadr[0]]
                         for j in JOINTS])

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
