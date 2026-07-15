"""TRELLIS 野菜 GLB → MuJoCo アセット変換パイプライン（D10 食材系統）。

GLB → 実寸スケール正規化 → 座標系整列（Z-up・底面 z=0・xy 中心原点）
    → 視覚メッシュ（デシメート + テクスチャ OBJ）→ CoACD 凸分解（衝突メッシュ群）
    → MJCF <asset>/<body> スニペット生成

出力: gyoza/assets/veg/<name>/{visual.obj, texture.png, collision_*.obj, snippet.xml}

重要（gyoza-research.md §3）: きゅうり丸ごとと輪切りセットは cut 差し替え用に
**同一スケール・同一整列規約**で処理する（SCALE を共有）。

実行: python scripts/veg_pipeline.py [--only mini_tomato] [--src ~/Downloads/vegetable-assets]
"""

from __future__ import annotations

import argparse
import pathlib

import numpy as np
import trimesh

REPO = pathlib.Path(__file__).resolve().parent.parent
OUT_ROOT = REPO / "gyoza" / "assets" / "veg"

CUCUMBER_LEN = 0.20  # 丸ごときゅうり全長 [m]。輪切りセットとスケール共有

# 実寸目標: (基準軸の実寸 [m], 基準軸 index) または cucumber 系は共有スケール値
ASSETS = {
    # ミニトマト: φ30mm（最優先・pick_place 用）
    "mini_tomato": {"target": (0.030, None)},  # None = 最大 extent を基準
    # きゅうり: 全長 200mm（x 軸が長軸）
    "whole_cucumber": {"scale": CUCUMBER_LEN / 1.0},
    # 輪切りセット: TRELLIS は各生成を独立正規化するため「同一スケール値」では整合しない。
    # 輪切り1枚の直径が丸ごとの直径(43.8mm)と一致するよう実測補正（丸ごとスケール比 43.8/90）
    "sliced_cucumber": {"scale": CUCUMBER_LEN / 1.0 * 43.8 / 90.0},
    # 輪切り単品: φ40mm（薄い軸 y を z へ立てる = 平置き）
    "sliced_cucumber_single": {"target": (0.040, None), "rot_x90": True},
    # ちぎりレタス: 長辺 90mm
    "torn_lettuce": {"target": (0.090, None)},
    # --- ラーメントッピング（2026-07-07、--src ~/Downloads/ramen-mesh で処理）---
    # 薄物 2 種は z_target で断面形状のまま押し出し（2026-07-08 ユーザー方針:
    # 「平面の形状をそのまま長く立体的に」= 把持寸法の確保。テクスチャ UV は不変）
    # 厚切りナルト: φ40mm、厚み 15.7 → 25mm（把持対象の本命）。GLB は Y-up → 平置きへ
    "naruto": {"target": (0.040, None), "rot_x90": True, "z_target": 0.025},
    # 厚切りチャーシュー: φ42mm、厚み → 22mm（把持・cut 演出対象）。
    # φ50 は把持不安定（smoke v3/v4）→ φ42 へ縮小。
    # 2026-07-08 診断（scratchpad/seating_test.py）: v5=0% の真因は「掴めるが可動ジョー上面に
    # 載って開放で外へ振り飛ばされる」。single_hull（凸包1個）は接近時に物体を弾き把持を悪化
    # させるだけで載り問題は直らない → False 維持。厚さ低減も把持を悪化させ効果なし。
    # 可動ジョー載りは球訓練 RL の把持過程に内在（アセット形状では解けない）。
    "chashu": {"target": (0.042, None), "rot_x90": True, "z_target": 0.022,
               "single_hull": False},
    # 味玉半切り: 長径 35mm（断面の向きは lineup 画像で要確認）
    "ajitama": {"target": (0.035, None), "rot_x90": True},
    # メンマ: 長辺 50mm（賑やかし）
    "menma": {"target": (0.050, None), "rot_x90": True},
    # --- 選別クープの果物（2026-07-10、--src ~/Downloads/fruits-mesh で処理）---
    # 訓練球 r16/r13 と径を一致させる（domrand ACT の把持分布内に収める）
    "shiratama": {"target": (0.032, None), "rot_x90": True},           # 白玉 φ32 (r16)
    # さくらんぼ: 最大 extent は茎込みのため、実の本体軸（最小の水平 extent）を φ26 に合わせる。
    # 初回の target(0.026,None) では本体 φ14-17mm となり、ジョー接近で弾き飛ばされ把持 0%
    # （smoke_grasp_cherry3、2026-07-10）。軸 index は rot_x90 後の extents 実測から選定
    "cherry_stage1_green": {"target": (0.026, 1), "rot_x90": True, "drop_small_hulls": 0.06},
    "cherry_stage2_yellowpink": {"target": (0.026, 0), "rot_x90": True, "drop_small_hulls": 0.06},
    "cherry_stage3_red": {"target": (0.026, 1), "rot_x90": True, "drop_small_hulls": 0.06},
    "grape": {"target": (0.026, None), "rot_x90": True},               # ぶどう φ26 (r13)
}

COACD_KW = dict(threshold=0.05, max_convex_hull=8, preprocess_mode="auto")
# 視覚メッシュはデシメートしない: fast_simplification が UV を落とし、最近傍継承は
# TRELLIS のテクスチャアトラス継ぎ目で破綻（ひび割れ状アーティファクト）するため。
# TRELLIS 出力 ~35万面でも MuJoCo の描画は問題ない（G1 実測 194 steps/s は別シーン要因）。
VISUAL_FACES = None


def load_mesh(glb: pathlib.Path) -> trimesh.Trimesh:
    scene = trimesh.load(str(glb))
    if isinstance(scene, trimesh.Scene):
        meshes = list(scene.geometry.values())
        assert len(meshes) == 1, f"{glb.name}: 複数ジオメトリ未対応"
        return meshes[0]
    return scene


def process(name: str, cfg: dict, src_dir: pathlib.Path, visual_only: bool = False):
    mesh = load_mesh(src_dir / f"{name}.glb")
    out = OUT_ROOT / name
    out.mkdir(parents=True, exist_ok=True)

    # --- 回転（GLB Y-up 系の薄物を Z-up 平置きへ）---
    if cfg.get("rot_x90"):
        mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))

    # --- スケール ---
    if "scale" in cfg:
        s = cfg["scale"]
    else:
        target, axis = cfg["target"]
        ext = mesh.extents
        s = target / (ext.max() if axis is None else ext[axis])
    mesh.apply_scale(s)

    # --- z 押し出し: 薄物の断面形状を保ったまま厚みを目標値へ（非等方スケール）---
    if "z_target" in cfg:
        sz = cfg["z_target"] / mesh.extents[2]
        mesh.apply_transform(np.diag([1.0, 1.0, sz, 1.0]))

    # --- 整列: xy 中心を原点、底面 z=0 ---
    lo, hi = mesh.bounds
    mesh.apply_translation([-(lo[0] + hi[0]) / 2, -(lo[1] + hi[1]) / 2, -lo[2]])

    # --- 視覚メッシュ: 元 UV を保持したまま OBJ 出力（デシメートしない — 上記コメント参照）---
    vis = mesh
    obj = trimesh.exchange.obj.export_obj(vis, include_texture=True, mtl_name="visual.mtl")
    (out / "visual.obj").write_text(obj)
    # テクスチャ画像
    img = getattr(mesh.visual.material, "baseColorTexture", None) or getattr(
        mesh.visual.material, "image", None)
    assert img is not None, f"{name}: テクスチャ画像が見つからない"
    img.save(out / "texture.png")

    # --- 衝突メッシュ: CoACD 凸分解（visual_only 時は既存出力を数えるだけ）---
    if visual_only:
        n_parts = len(list(out.glob("collision_*.obj")))
        assert n_parts > 0, f"{name}: 既存 collision が無いのに --visual-only"
        parts = range(n_parts)
    elif cfg.get("single_hull"):
        # 単一凸包: 表面起伏を消してジョーとの機械的インターロックを防ぐ（把持安定優先）。
        # CoACD は走らせない — 凸包1個のみ出力する（走らせると凹みを復元し谷が戻るため）。
        for p in out.glob("collision_*.obj"):
            p.unlink()
        mesh.convex_hull.export(out / "collision_0.obj")
        parts = [None]  # 1 hull
    else:
        # 通常: CoACD 凸分解（凹み形状を凸包群で近似）
        for p in out.glob("collision_*.obj"):
            p.unlink()
        import coacd
        cm = coacd.Mesh(mesh.vertices, mesh.faces)
        parts = coacd.run_coacd(cm, **COACD_KW)
        # drop_small_hulls: 体積比が閾値未満の凸包を衝突から除外（さくらんぼの茎対策。
        # 細長い茎凸包にジョーが引っかかると実全体が弾き飛ばされる — smoke_grasp_cherry3。
        # 視覚メッシュは茎付きのまま、物理は実の本体のみにする）
        frac = cfg.get("drop_small_hulls")
        if frac:
            hulls = [trimesh.Trimesh(v, f) for v, f in parts]
            vols = [h.convex_hull.volume for h in hulls]
            keep = [h for h, v in zip(hulls, vols) if v >= frac * sum(vols)]
            print(f"[{name}] drop_small_hulls: {len(hulls)} -> {len(keep)} hulls "
                  f"(vol fracs={[round(v / sum(vols), 3) for v in vols]})")
            parts = keep
            for i, h in enumerate(keep):
                h.export(out / f"collision_{i}.obj")
        else:
            for i, (v, f) in enumerate(parts):
                trimesh.Trimesh(v, f).export(out / f"collision_{i}.obj")

    # --- MJCF スニペット ---
    prefix = name
    lines = [f'<!-- {name}: scale={s:.5f}, 底面 z=0, 視覚 {len(vis.faces)} faces, 衝突 {len(parts)} hulls -->',
             "<asset>",
             f'  <texture type="2d" name="{prefix}_tex" file="veg/{name}/texture.png"/>',
             f'  <material name="{prefix}_mat" texture="{prefix}_tex" specular="0.1" shininess="0.1"/>',
             f'  <mesh name="{prefix}_visual" file="veg/{name}/visual.obj"/>']
    for i in range(len(parts)):
        lines.append(f'  <mesh name="{prefix}_col{i}" file="veg/{name}/collision_{i}.obj"/>')
    lines += ["</asset>", "<body>",
              f'  <geom type="mesh" mesh="{prefix}_visual" material="{prefix}_mat" class="visual"/>']
    for i in range(len(parts)):
        lines.append(f'  <geom type="mesh" mesh="{prefix}_col{i}" group="3"/>')
    lines.append("</body>")
    (out / "snippet.xml").write_text("\n".join(lines))

    ext = mesh.extents
    print(f"[{name}] scale={s:.5f} extents(mm)={np.round(ext * 1000, 1).tolist()} "
          f"visual_faces={len(vis.faces)} hulls={len(parts)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="~/Downloads/vegetable-assets")
    ap.add_argument("--only", default=None)
    ap.add_argument("--visual-only", action="store_true",
                    help="CoACD を再実行せず視覚メッシュと snippet のみ再生成")
    args = ap.parse_args()
    src = pathlib.Path(args.src).expanduser()
    for name, cfg in ASSETS.items():
        if args.only and name not in args.only.split(","):
            continue
        process(name, cfg, src, visual_only=args.visual_only)


if __name__ == "__main__":
    main()
