"""ラーメン丼シーン: 実験用の平皿 body をランタイムで丼へ差し替える（単一 XML 維持）。

デモ専用（実験の平皿 pick_place.xml は無改修）。スープ面 z を現皿天面(0.012)に揃え、
ハイブリッド expert / ゴール条件付き ACT を無改修で移植できるようにする（HANDOFF 方針）。

構成（MuJoCo にお椀プリミティブが無いため primitives で近似）:
  - ceramic_base: 白陶器の底円盤（衝突あり・構造）
  - soup: スープ面の平円盤（衝突あり・トッピングはここに載る。天面 z=0.012）
  - wall_*: 外側へ傾けた box リング = 台形（下すぼまり・上広がり）の丼壁（視覚のみ contype=0）
  - rimline_*: ふちの赤ライン（視覚のみ）

寸法方針（2026-07-08 ユーザーFB反映）: 現実の丼は上下逆の台形。壁を外側に傾けて再現。
外径は到達可能ディスク（半径~1.8cm）に合わせて縮小（旧 17cm → ~10cm）。壁は contype=0 で
アーム/カメラを妨げず、スープ面 z=0.012 は据え置き（expert 無改修）。
"""
from __future__ import annotations

import numpy as np
import mujoco


def patch_bowl(spec: "mujoco.MjSpec", center_xy=(0.204, -0.118),
               foot_r: float = 0.036, rim_r: float = 0.065, wall_h: float = 0.042,
               soup_z: float = 0.030, n_seg: int = 48,
               show_zone: bool = False, zone_r: float = 0.055) -> None:
    """plate body を台形の丼へ差し替える。center_xy に body を移設。

    foot_r: 底（テーブル接地）半径、rim_r: ふち（上端）半径、wall_h: 壁の高さ。
    foot_r < rim_r で外側に開いた台形断面になる。soup_z: スープ面天面（既定 0.012）。
    show_zone=True: check_success() の xy 成功ゾーン（半径 zone_r）を半透明ディスクで可視化。
    """
    def ensure_mat(name, rgba, spec_kw):
        if spec.material(name) is not None:   # 未存在時は None を返す（例外でない）
            return
        m = spec.add_material(name=name)
        m.rgba = rgba
        for k, v in spec_kw.items():
            setattr(m, k, v)
    ensure_mat("ceramic_mat", [0.96, 0.96, 0.94, 1], dict(specular=0.35, shininess=0.5))
    ensure_mat("soup_mat", [0.80, 0.52, 0.30, 1], dict(specular=0.15, shininess=0.2))
    ensure_mat("rim_accent_mat", [0.62, 0.14, 0.12, 1], dict(specular=0.2, shininess=0.3))

    body = spec.body("plate")
    body.pos = [center_xy[0], center_xy[1], 0.0]
    for g in list(body.geoms):          # 既存の平皿 geom を除去
        spec.delete(g)

    # スープ面が壁と接する半径（壁を z=soup_z で内挿）
    soup_r = foot_r + (rim_r - foot_r) * (soup_z / wall_h)

    # 底（衝突・構造）: soup 面直下を埋める低い円盤（soup より僅かに小径・低く=白縁を出さない）
    body.add_geom(name="ceramic_base", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                  size=[soup_r - 0.001, (soup_z - 0.002) / 2, 0],
                  pos=[0, 0, (soup_z - 0.002) / 2], material="ceramic_mat")
    # スープ面（衝突・トッピング着地面）: 天面 z=soup_z
    body.add_geom(name="soup", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                  size=[soup_r, 0.004, 0], pos=[0, 0, soup_z - 0.004], material="soup_mat")

    # 台形の壁 = 外側へ傾けた box を回転体状に並べる（視覚のみ・contype=0）。
    # 各 box: 底(foot_r, z=0) → 上端(rim_r, z=wall_h) の斜め壁。
    # 向き = Rz(方位) ⊗ Ry(壁の傾き)。Ry で長軸を Z から外向き(+r)へ倒す。
    r_mid, z_mid = (foot_r + rim_r) / 2, wall_h / 2
    wall_len = float(np.hypot(rim_r - foot_r, wall_h))     # 斜め壁の長さ
    tilt = float(np.arctan2(rim_r - foot_r, wall_h))       # 鉛直からの傾き
    ct, st = np.cos(tilt / 2), np.sin(tilt / 2)
    seg_w = np.pi * r_mid / n_seg * 1.35                   # 隣接 box が重なる幅
    for i in range(n_seg):
        a = 2 * np.pi * i / n_seg
        ca, sa = np.cos(a / 2), np.sin(a / 2)
        quat = [ca * ct, -sa * st, ca * st, sa * ct]       # Rz(a)⊗Ry(tilt)
        body.add_geom(name=f"wall_{i}", type=mujoco.mjtGeom.mjGEOM_BOX,
                      size=[0.0035, seg_w, wall_len / 2],
                      pos=[r_mid * np.cos(a), r_mid * np.sin(a), z_mid],
                      quat=quat, material="ceramic_mat", contype=0, conaffinity=0)
    # ふちの赤ライン（上端の内周・視覚のみ・細め）
    for i in range(n_seg):
        a = 2 * np.pi * i / n_seg
        seg_w2 = np.pi * rim_r / n_seg * 1.35
        body.add_geom(name=f"rimline_{i}", type=mujoco.mjtGeom.mjGEOM_BOX,
                      size=[0.0026, seg_w2, 0.0015],
                      pos=[rim_r * np.cos(a), rim_r * np.sin(a), wall_h - 0.0012],
                      quat=[np.cos(a / 2), 0, 0, np.sin(a / 2)],
                      material="rim_accent_mat", contype=0, conaffinity=0)

    # 成功判定ゾーン（xy 半径）の可視化（視覚のみ・目視確認用。デモ本番は show_zone=False）
    if show_zone:
        ensure_mat("zone_mat", [0.20, 0.90, 0.35, 0.28], dict(specular=0.0, shininess=0.0))
        body.add_geom(name="success_zone", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                      size=[zone_r, 0.0006, 0], pos=[0, 0, soup_z + 0.001],
                      material="zone_mat", contype=0, conaffinity=0)

    # ゾーン判定用 site は soup 天面へ
    try:
        spec.site("plate_center").pos = [0, 0, soup_z]
    except Exception:
        pass


def patch_coupe(spec: "mujoco.MjSpec", center_xy=(0.204, -0.118),
                rim_r: float = 0.062, floor_r: float = 0.045, floor_z: float = 0.058,
                bowl_depth: float = 0.030, foot_r: float = 0.034, foot_h: float = 0.004,
                stem_r: float = 0.0045, n_seg: int = 48) -> float:
    """plate body をクラシックなクープグラス（脚付き・浅い広口）へ差し替える。

    参考画像: media/ ユーザー提供のシャンパンクープ（台座 + 細い脚 + 浅いソーサー型ボウル）。
    設計制約（HANDOFF ★節）:
      - 内底は平底必須（衝突あり・着地面。曲面底だと球が中央に転がり配置の意味が消える）
      - 壁は透明ガラス（rgba alpha）。最下段リングのみ衝突を持たせ球の転がり出しを防ぐ
      - 成功判定 site / ゾーン z は内底 floor_z に合わせる（呼び出し側は返り値で zone_z を調整）
    返り値: floor_z（内底天面の z）
    """
    def ensure_mat(name, rgba, spec_kw):
        if spec.material(name) is not None:
            return
        m = spec.add_material(name=name)
        m.rgba = rgba
        for k, v in spec_kw.items():
            setattr(m, k, v)
    ensure_mat("glass_wall_mat", [0.82, 0.89, 0.93, 0.25], dict(specular=0.7, shininess=0.85))
    ensure_mat("glass_body_mat", [0.85, 0.91, 0.94, 0.45], dict(specular=0.6, shininess=0.8))
    ensure_mat("glass_floor_mat", [0.88, 0.93, 0.95, 0.60], dict(specular=0.5, shininess=0.7))

    body = spec.body("plate")
    body.pos = [center_xy[0], center_xy[1], 0.0]
    for g in list(body.geoms):
        spec.delete(g)

    # 台座（衝突あり）+ 細い脚（視覚のみ）
    body.add_geom(name="coupe_foot", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                  size=[foot_r, foot_h / 2, 0], pos=[0, 0, foot_h / 2],
                  material="glass_body_mat")
    stem_top = floor_z - 0.006  # ボウル下面まで
    body.add_geom(name="coupe_stem", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                  size=[stem_r, (stem_top - foot_h) / 2, 0],
                  pos=[0, 0, foot_h + (stem_top - foot_h) / 2],
                  material="glass_body_mat", contype=0, conaffinity=0)
    # 内底 = 平底の着地面（衝突あり・天面 z=floor_z）
    body.add_geom(name="coupe_floor", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                  size=[floor_r, 0.003, 0], pos=[0, 0, floor_z - 0.003],
                  material="glass_floor_mat")

    # ソーサー型の湾曲壁を3段の円錐リング（box 回転体）で近似。
    # プロフィール: (floor_r, floor_z) → 大きく開く → ふち付近はほぼ鉛直。
    prof = [
        (floor_r, floor_z, rim_r - 0.006, floor_z + bowl_depth * 0.55),   # 下段（開く）
        (rim_r - 0.006, floor_z + bowl_depth * 0.55, rim_r, floor_z + bowl_depth * 0.85),  # 中段
        (rim_r, floor_z + bowl_depth * 0.85, rim_r, floor_z + bowl_depth),  # 上段（鉛直）
    ]
    for k, (r0, z0, r1, z1) in enumerate(prof):
        r_mid, z_mid = (r0 + r1) / 2, (z0 + z1) / 2
        wall_len = float(np.hypot(r1 - r0, z1 - z0))
        tilt = float(np.arctan2(r1 - r0, z1 - z0))
        ct, st = np.cos(tilt / 2), np.sin(tilt / 2)
        seg_w = np.pi * r_mid / n_seg * 1.35
        collide = (k == 0)  # 最下段のみ衝突（球の転がり出し防止バリア）
        for i in range(n_seg):
            a = 2 * np.pi * i / n_seg
            ca, sa = np.cos(a / 2), np.sin(a / 2)
            quat = [ca * ct, -sa * st, ca * st, sa * ct]
            body.add_geom(name=f"coupe_wall{k}_{i}", type=mujoco.mjtGeom.mjGEOM_BOX,
                          size=[0.0018, seg_w, wall_len / 2],
                          pos=[r_mid * np.cos(a), r_mid * np.sin(a), z_mid],
                          quat=quat, material="glass_wall_mat",
                          contype=1 if collide else 0, conaffinity=1 if collide else 0)

    try:
        spec.site("plate_center").pos = [0, 0, floor_z]
    except Exception:
        pass
    return floor_z
