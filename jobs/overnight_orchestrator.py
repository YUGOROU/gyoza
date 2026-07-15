# /// script
# requires-python = ">=3.11"
# dependencies = ["huggingface_hub"]
# ///
"""オーバーナイト・オーケストレータ（2026-07-11）: datagen v2 → merge → ACT 学習 → eval を
3果物並行で完走させる。HF Jobs cpu-basic 上で動き、子ジョブを入れ子起動する
（入れ子起動・secrets 伝播は job 6a511431/6a51143d で検証済み）。

datagen v2 の設計（2026-07-10 ablation 診断の反映）:
  - 距離物なし（旧 GYOZA_DISTRACT_MESH を渡さない）= 測定側 MASK_STATICS=1 の
    「ライブのみ」ビューと訓練分布を構成的に一致させる
  - GYOZA_COUPE=1（前回の渡し忘れを是正、eval/本番と同一 env）
  - 層化サンプリング（CELLS 4x4 グリッド・セル毎 keep 目標）= 奥側デッドゾーンの根治

ゲート（チェッカー兼ブロッカー）:
  - datagen: 総 keep >= KEEP_OK(160) → merge / >= KEEP_MIN(100) → 拡張シャード1本 → merge
    / 未満 → その果物は停止（他果物は継続）
  - train: ERROR なら1回リトライ
  - eval: 2シード。結果は summary に記録するのみ

実行:
    hf jobs uv run --flavor cpu-basic --timeout 10h --detach --secrets HF_TOKEN \
        --label name=overnight-orchestrator --label project=gyoza \
        -v hf://buckets/YUGOROU/gyoza-sim:/gyoza --env GYOZA_DATA=/gyoza \
        --env RUN_TAG=0711 jobs/overnight_orchestrator.py

env vars: RUN_TAG(データセット/出力の接尾辞) / KEEP_OK(160) / KEEP_MIN(100) / STEPS(100000)
進捗: bucket outputs/overnight_<RUN_TAG>/status.json を随時更新。
"""

import json
import math
import os
import pathlib
import re
import subprocess
import threading
import time

DATA = pathlib.Path(os.environ.get("GYOZA_DATA", "/gyoza"))
JOBS = DATA / "code" / "jobs"
RUN_TAG = os.environ.get("RUN_TAG", "0711")
KEEP_OK = int(os.environ.get("KEEP_OK", "160"))
KEEP_MIN = int(os.environ.get("KEEP_MIN", "100"))
STEPS = os.environ.get("STEPS", "100000")
OUT = DATA / "outputs" / f"overnight_{RUN_TAG}"

BUCKET = "hf://buckets/YUGOROU/gyoza-sim:/gyoza"
COMMON = ["--secrets", "HF_TOKEN", "-v", BUCKET, "--env", "GYOZA_DATA=/gyoza",
          "--label", "project=gyoza"]

FRUITS = {
    # key: (GYOZA_TOMATO 名, policy repo, keep_per_cell, max_att_per_cell)
    "shiratama": ("shiratama", "YUGOROU/act_gyoza_shiratama_v2", 14, 30),
    "grape": ("grape", "YUGOROU/act_gyoza_grape_v2", 14, 60),
    "cherry3": ("cherry_stage3_red", "YUGOROU/act_gyoza_cherry3_v2", 14, 70),
}

_lock = threading.Lock()
_status = {"run_tag": RUN_TAG, "started": time.strftime("%Y-%m-%d %H:%M:%S"), "fruits": {}}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def write_status(fruit=None, **kv):
    with _lock:
        if fruit:
            _status["fruits"].setdefault(fruit, {}).update(kv)
        _status["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
        OUT.mkdir(parents=True, exist_ok=True)  # bucket FUSE は空 dir を保持しない → 毎回
        (OUT / "status.json").write_text(json.dumps(_status, indent=2, ensure_ascii=False))


def sh(cmd, timeout=600):
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def submit(script, flavor, timeout, name, env: dict, retries=3):
    cmd = ["hf", "jobs", "uv", "run", "--detach", "--flavor", flavor,
           "--timeout", timeout, "--label", f"name={name}"] + COMMON
    for k, v in env.items():
        cmd += ["--env", f"{k}={v}"]
    cmd.append(str(JOBS / script))
    for i in range(retries):
        r = sh(cmd)
        m = re.search(r"id[=:]\s*([0-9a-f]{20,})", r.stdout)
        if r.returncode == 0 and m:
            log(f"submit {name} -> {m.group(1)}")
            return m.group(1)
        log(f"submit {name} failed (try {i}): {r.stdout[-200:]} {r.stderr[-200:]}")
        time.sleep(30)
    return None


def stage_of(job_id):
    # CLI 出力パースは Job 内で不安定（初回オーケストレータが wait_jobs で無限待機した
    # 実障害 2026-07-11）→ huggingface_hub Python API を使う
    from huggingface_hub import inspect_job
    try:
        return inspect_job(job_id=job_id).status.stage
    except Exception as e:
        log(f"inspect {job_id} failed: {e}")
        return "?"


def wait_jobs(ids, timeout_s, poll=90):
    """全ジョブの終了を待つ。{id: stage} を返す（タイムアウト分は最後の stage のまま）。"""
    deadline = time.time() + timeout_s
    stages = {i: "RUNNING" for i in ids if i}
    while time.time() < deadline:
        for i in list(stages):
            if stages[i] not in ("COMPLETED", "ERROR", "CANCELED"):
                stages[i] = stage_of(i)
        if all(s in ("COMPLETED", "ERROR", "CANCELED") for s in stages.values()):
            break
        time.sleep(poll)
    return stages


def logs_of(job_id):
    from huggingface_hub import fetch_job_logs
    try:
        return "\n".join(fetch_job_logs(job_id=job_id))
    except Exception as e:
        log(f"logs {job_id} failed: {e}")
        return ""


def parse_done_json(text, marker):
    m = None
    for m in re.finditer(re.escape(marker) + r"\s+(\{.*\})", text):
        pass
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            return None
    return None


def datagen_env(fruit_key, cells, keep_per_cell, max_att, repo, seed):
    name, _, _, _ = FRUITS[fruit_key]
    return {"MODEL": "outputs/rl_v3a-warm-perf/best/best_model.zip",
            "REPO_ID": repo, "GYOZA_TOMATO": name, "GYOZA_COUPE": "1",
            "CELLS": ",".join(str(c) for c in cells),
            "KEEP_PER_CELL": str(keep_per_cell), "MAX_ATT_PER_CELL": str(max_att),
            "SEED": str(seed)}


def run_fruit(fruit_key):
    name, policy_repo, kpc, max_att = FRUITS[fruit_key]
    ds_prefix = f"YUGOROU/gyoza-fruit2-{fruit_key}-{RUN_TAG}"
    write_status(fruit_key, phase="datagen", policy=policy_repo)

    # --- 1) datagen 4シャード（各4セル・インターリーブ割り） ---
    # PREBAKED: outputs/overnight_<RUN_TAG>/prebaked_keeps.json があれば datagen 済みとして
    # スキップ（再開モード。初回オーケストレータ障害からの復帰用）
    prebaked = OUT / "prebaked_keeps.json"
    if prebaked.exists():
        keeps = json.loads(prebaked.read_text()).get(fruit_key, {})
        log(f"{fruit_key}: PREBAKED keeps={keeps}")
    else:
        shard_ids, shard_repos = [], []
        for k in range(4):
            cells = list(range(k, 16, 4))
            repo = f"{ds_prefix}-p{k + 1}"
            jid = submit("act_datagen_job.py", "t4-small", "2h",
                         f"dg2-{fruit_key}-p{k + 1}",
                         datagen_env(fruit_key, cells, kpc, max_att, repo, seed=80 + k))
            shard_ids.append(jid)
            shard_repos.append(repo)
        write_status(fruit_key, datagen_jobs=shard_ids)
        wait_jobs(shard_ids, timeout_s=110 * 60)
        keeps = {}
        for jid, repo in zip(shard_ids, shard_repos):
            d = parse_done_json(logs_of(jid), "[datagen] DONE") if jid else None
            keeps[repo] = (d or {}).get("kept", 0)
    total_keep = sum(keeps.values())
    good_repos = [r for r, k in keeps.items() if k > 0]
    log(f"{fruit_key}: datagen keeps={keeps} total={total_keep}")
    write_status(fruit_key, datagen_keeps=keeps, total_keep=total_keep)

    # --- 2) keep ゲート ---
    if total_keep < KEEP_MIN:
        write_status(fruit_key, phase="ABORTED_low_keep")
        log(f"{fruit_key}: ABORT (keep {total_keep} < {KEEP_MIN})")
        return
    if total_keep < KEEP_OK:
        deficit = KEEP_OK - total_keep
        ext_kpc = max(1, math.ceil(deficit / 16))
        repo = f"{ds_prefix}-ext"
        write_status(fruit_key, phase="datagen_ext", ext_kpc=ext_kpc)
        jid = submit("act_datagen_job.py", "t4-small", "2h", f"dg2-{fruit_key}-ext",
                     datagen_env(fruit_key, list(range(16)), ext_kpc, 30, repo, seed=99))
        wait_jobs([jid], timeout_s=100 * 60)
        d = parse_done_json(logs_of(jid), "[datagen] DONE") if jid else None
        ext_kept = (d or {}).get("kept", 0)
        total_keep += ext_kept
        if ext_kept > 0:
            good_repos.append(repo)
        write_status(fruit_key, ext_kept=ext_kept, total_keep=total_keep)
        if total_keep < KEEP_MIN:
            write_status(fruit_key, phase="ABORTED_low_keep")
            return

    # --- 3) merge（入力が1本ならスキップしてそのまま学習へ） ---
    if len(good_repos) == 1:
        merged = good_repos[0]
        write_status(fruit_key, phase="merge_skipped", merged_repo=merged)
    else:
        merged = f"{ds_prefix}-v1"
        write_status(fruit_key, phase="merge", merged_repo=merged)
        jid = submit("merge_datasets_job.py", "cpu-upgrade", "1h", f"merge2-{fruit_key}",
                     {"REPO_IDS": ",".join(good_repos), "NEW_REPO_ID": merged})
        stages = wait_jobs([jid], timeout_s=45 * 60)
        if not jid or stages.get(jid) != "COMPLETED":
            write_status(fruit_key, phase="ABORTED_merge_failed")
            return

    # --- 4) ACT 学習（ERROR は1回リトライ） ---
    for attempt in range(2):
        write_status(fruit_key, phase=f"train_att{attempt}")
        jid = submit("act_train_job.py", "a100-large", "4h", f"train2-{fruit_key}",
                     {"DATASET": merged, "POLICY_REPO": policy_repo,
                      "RUN": f"act_{fruit_key}_v2", "STEPS": STEPS})
        stages = wait_jobs([jid], timeout_s=220 * 60)
        if jid and stages.get(jid) == "COMPLETED":
            break
        log(f"{fruit_key}: train attempt {attempt} -> {stages}")
    else:
        write_status(fruit_key, phase="ABORTED_train_failed")
        return

    # --- 5) eval 2シード（クープ・距離物なし = マスク運用ビューと同一条件） ---
    write_status(fruit_key, phase="eval")
    eval_ids, eval_runs = [], []
    for seed in (42, 43):
        run = f"eval_{fruit_key}_v2_s{seed}"
        jid = submit("act_eval_job.py", "t4-small", "1h", f"ev2-{fruit_key}-s{seed}",
                     {"POLICY": policy_repo, "GYOZA_TOMATO": name, "GYOZA_COUPE": "1",
                      "EPISODES": "25", "SEED": str(seed), "RUN": run, "VIDEO_N": "3"})
        eval_ids.append(jid)
        eval_runs.append(run)
    wait_jobs(eval_ids, timeout_s=50 * 60)
    results = {}
    for run in eval_runs:
        p = DATA / "outputs" / "evals" / run / "stats.json"
        try:
            results[run] = json.loads(p.read_text()).get("success_rate")
        except Exception:
            results[run] = None
    write_status(fruit_key, phase="DONE", eval=results)
    log(f"{fruit_key}: DONE eval={results}")


def main():
    write_status()
    log(f"orchestrator start RUN_TAG={RUN_TAG}")
    # 旧ポリシーのベースライン eval（新条件: クープ・距離物なし）を独立に投入
    base_ids = []
    for key, (name, _, _, _) in FRUITS.items():
        old_policy = {"shiratama": "YUGOROU/act_gyoza_shiratama",
                      "grape": "YUGOROU/act_gyoza_grape",
                      "cherry3": "YUGOROU/act_gyoza_cherry3"}[key]
        jid = submit("act_eval_job.py", "t4-small", "1h", f"ev-base-{key}",
                     {"POLICY": old_policy, "GYOZA_TOMATO": name, "GYOZA_COUPE": "1",
                      "EPISODES": "25", "SEED": "42", "RUN": f"eval_{key}_old_clean", "VIDEO_N": "0"})
        base_ids.append((key, jid))

    threads = [threading.Thread(target=run_fruit, args=(k,), name=k) for k in FRUITS]
    for t in threads:
        t.start()
        time.sleep(5)
    for t in threads:
        t.join()

    # ベースライン回収
    wait_jobs([j for _, j in base_ids], timeout_s=10 * 60)
    for key, jid in base_ids:
        p = DATA / "outputs" / "evals" / f"eval_{key}_old_clean" / "stats.json"
        try:
            write_status(key, baseline_old_clean=json.loads(p.read_text()).get("success_rate"))
        except Exception:
            write_status(key, baseline_old_clean=None)

    write_status()
    log("orchestrator DONE " + json.dumps(_status["fruits"], ensure_ascii=False))


if __name__ == "__main__":
    main()
