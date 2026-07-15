"""判定 VLM バリアントのオフライン評価（プロトコル v3 凍結用）。

judge_bench_job.py が生成した GT ラベル付き before/after クロップペアに対し、
判定器バリアントを比較する。sim 不要・API 呼び出しのみなのでローカル実行可。

使い方:
    .venv/bin/python scripts/judge_bench_eval.py <bench_dir> [--variant V0 V1 ...] [--model M]

バリアント:
    V0: 現行 = 2枚比較カウント1コール（ablation v2 で FN 49% の再現確認用）
    V1: 単一画像カウント×2コール（before/after 独立、差分はコード側）
    V2: V1 + クロップ2倍拡大
    V3: V2 + 色弁別ヒント付きプロンプト
"""

import argparse
import base64
import io
import json
import pathlib
import re
import sys
import time

from huggingface_hub import InferenceClient, get_token
from PIL import Image

LABEL = {"shiratama": "white shiratama dumpling", "grape": "purple grape",
         "cherry_stage3_red": "ripe red cherry"}
HINT = {
    "shiratama": "Shiratama are matte WHITE spheres. Do not count purple grapes or red cherries.",
    "grape": "Grapes are DARK PURPLE, almost black, small spheres. Do not count red cherries "
             "(bright red) or white shiratama. Grapes may be hard to see against the glass.",
    "cherry_stage3_red": "Ripe cherries are BRIGHT RED. Do not count dark purple grapes "
                         "(darker, almost black) or white shiratama.",
}


def png_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def call(client, model, content, max_tokens=8192, retries=5):
    """空応答もリトライ対象。全滅時は None（呼び手がエラーとして扱う）。"""
    for attempt in range(retries):
        try:
            r = client.chat.completions.create(
                model=model, messages=[{"role": "user", "content": content}],
                max_tokens=max_tokens)
            text = (r.choices[0].message.content or "").strip()
            if text:
                return text
        except Exception as e:
            print(f"    (api error att{attempt}: {str(e)[:80]})", file=sys.stderr)
        time.sleep(min(4 * 2 ** attempt, 60))
    return None


def parse_last_int(text):
    nums = re.findall(r"\d+", text)
    return int(nums[-1]) if nums else None


def judge_v0(client, model, before, after, fruit):
    """現行: 2枚比較カウント1コール。"""
    prompt = (f"Two close-up top-down photos of the SAME glass coupe. FIRST=before, "
              f"SECOND=after. Count the number of {LABEL[fruit]}s that are INSIDE the coupe "
              f"in each photo. Answer with exactly two integers separated by a space: "
              f"<before_count> <after_count>")
    text = call(client, model, [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{png_b64(before)}"}},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{png_b64(after)}"}},
    ])
    if text is None:
        return None, "(api exhausted)"
    nums = re.findall(r"\d+", text)
    if len(nums) < 2:
        return None, text
    return int(nums[-1]) > int(nums[-2]), text


def count_single(client, model, img, fruit, hint=False):
    prompt = (f"Close-up top-down photo of a glass coupe. Count the {LABEL[fruit]}s that are "
              f"INSIDE the coupe. {HINT[fruit] if hint else ''} "
              f"End your answer with a single line: COUNT=<integer>")
    text = call(client, model, [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{png_b64(img)}"}},
    ])
    if text is None:
        return None, "(api exhausted)"
    m = re.search(r"COUNT\s*=\s*(\d+)", text)
    n = int(m.group(1)) if m else parse_last_int(text)
    return n, text


def judge_split(client, model, before, after, fruit, upscale=1, hint=False):
    """単一画像カウント×2コール、差分はコード側。"""
    if upscale > 1:
        before = before.resize((before.width * upscale, before.height * upscale), Image.LANCZOS)
        after = after.resize((after.width * upscale, after.height * upscale), Image.LANCZOS)
    nb, tb = count_single(client, model, before, fruit, hint)
    na, ta = count_single(client, model, after, fruit, hint)
    if nb is None or na is None:
        return None, f"b:{tb[:60]} / a:{ta[:60]}"
    return na > nb, f"b={nb} a={na}"


VARIANTS = {
    "V0": lambda c, m, b, a, f: judge_v0(c, m, b, a, f),
    "V1": lambda c, m, b, a, f: judge_split(c, m, b, a, f),
    "V2": lambda c, m, b, a, f: judge_split(c, m, b, a, f, upscale=2),
    "V3": lambda c, m, b, a, f: judge_split(c, m, b, a, f, upscale=2, hint=True),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bench_dir")
    ap.add_argument("--variant", nargs="+", default=["V0", "V1", "V2", "V3"])
    ap.add_argument("--model", default="moonshotai/Kimi-K2.6")
    ap.add_argument("--limit", type=int, default=0, help="ペア数上限（0=全部）")
    args = ap.parse_args()

    bench = pathlib.Path(args.bench_dir)
    labels = json.loads((bench / "labels.json").read_text())
    if args.limit:
        labels = labels[:args.limit]
    client = InferenceClient(api_key=get_token(), provider="auto")

    report = {}
    for var in args.variant:
        fn = VARIANTS[var]
        stats = dict(tp=0, tn=0, fp=0, fn=0, err=0)
        by_fruit = {}
        details = []
        for lab in labels:
            before = Image.open(bench / f"{lab['id']}_before.png")
            after = Image.open(bench / f"{lab['id']}_after.png")
            verdict, raw = fn(client, args.model, before, after, lab["fruit"])
            gt = lab["added"]
            if verdict is None:
                key = "err"
            elif verdict and gt:
                key = "tp"
            elif not verdict and not gt:
                key = "tn"
            elif verdict and not gt:
                key = "fp"
            else:
                key = "fn"
            stats[key] += 1
            bf = by_fruit.setdefault(lab["fruit"], dict(tp=0, tn=0, fp=0, fn=0, err=0))
            bf[key] += 1
            details.append(dict(id=lab["id"], fruit=lab["fruit"], gt=gt,
                                verdict=verdict, raw=raw[:150], outcome=key))
            print(f"[{var} {lab['id']}] {lab['fruit']} gt={gt} -> {verdict} ({key})", flush=True)
        n = len(labels)
        acc = (stats["tp"] + stats["tn"]) / n
        pos = stats["tp"] + stats["fn"]
        neg = stats["tn"] + stats["fp"]
        print(f"\n=== {var} ({args.model}) acc={acc:.1%} "
              f"FN={stats['fn']}/{pos} FP={stats['fp']}/{neg} err={stats['err']}\n"
              f"    by fruit: {by_fruit}\n", flush=True)
        report[var] = dict(model=args.model, stats=stats, by_fruit=by_fruit,
                           acc=acc, details=details)

    out = bench / "eval_report.json"
    existing = json.loads(out.read_text()) if out.exists() else {}
    existing.update({f"{args.model}:{k}": v for k, v in report.items()})
    out.write_text(json.dumps(existing, indent=1))
    print(f"[report] -> {out}")


if __name__ == "__main__":
    main()
