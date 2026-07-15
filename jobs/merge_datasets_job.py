# /// script
# requires-python = ">=3.11"
# dependencies = ["lerobot==0.4.4"]
# ///
"""複数 LeRobotDataset を lerobot 純正 `lerobot-edit-dataset merge` で1本に統合。

datagen を seed 分割で並列生成した際の後処理（lerobot 0.4.4 は学習時に
--dataset.repo_id を複数取れない = make_dataset が単一 str 前提。学習前にマージ必須）。
merge は動画特徴に対応（同一 vcodec なら再エンコードなしで集約）。--repo_id は merge では
無視され、--operation.repo_ids に列挙した全 repo が結合対象。

実行:
    hf jobs uv run --flavor cpu-upgrade --timeout 1h --detach --secrets HF_TOKEN \
        --env REPO_IDS=YUGOROU/gyoza-naruto-goal-synth-p1,YUGOROU/gyoza-naruto-goal-synth-p2,YUGOROU/gyoza-naruto-goal-synth-p3 \
        --env NEW_REPO_ID=YUGOROU/gyoza-naruto-goal-synth-v2 \
        jobs/merge_datasets_job.py

env vars: REPO_IDS(カンマ区切り) / NEW_REPO_ID
"""

import os
import subprocess
import sys


def main():
    repo_ids = [s.strip() for s in os.environ["REPO_IDS"].split(",") if s.strip()]
    new_repo_id = os.environ["NEW_REPO_ID"]
    # draccus は List を Python リテラル文字列で受ける（"['a', 'b']"）。
    repo_ids_lit = "[" + ", ".join(f"'{r}'" for r in repo_ids) + "]"
    # lerobot 0.4.4 handle_merge の実仕様（ソース確認済み・docstring/deepwiki は誤り）:
    #   --repo_id = 出力先 repo（output_dir = HF_LEROBOT_HOME/repo_id）
    #   --operation.repo_ids = 統合する入力データセット群
    #   --new_repo_id は merge では未使用
    # 出力 repo_id は入力のどれとも別名にすること（同名だと入力 DL と出力 dir が衝突し
    # FileExistsError）。
    cmd = [
        "lerobot-edit-dataset",
        "--repo_id", new_repo_id,
        "--operation.type", "merge",
        "--operation.repo_ids", repo_ids_lit,
        "--push_to_hub", "true",
    ]
    print("[merge]", " ".join(cmd), flush=True)
    r = subprocess.run(cmd)
    if r.returncode != 0:
        sys.exit(r.returncode)
    print("[merge] DONE", {"merged": repo_ids, "into": new_repo_id}, flush=True)


if __name__ == "__main__":
    main()
