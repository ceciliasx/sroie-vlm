"""
Score a VLM (Claude) on SROIE key-field extraction: image -> {company, date,
address, total} directly, no OCR step. Compare against eval_ocr.py's
Tesseract+regex pipeline on the same metric once that exists.

Setup:
    pip install anthropic
    setx ANTHROPIC_API_KEY "sk-..."     (new terminal after this)

Usage:
    python eval_vlm.py --limit 10        # cheap sanity check first
    python eval_vlm.py --limit 50 --model claude-haiku-5-5
    python eval_vlm.py                   # all 347 test images (costs money - check pricing first)

Results are cached in results/vlm/<model>/<id>.json, so a re-run only calls
the API for images that aren't done yet.
"""
import argparse
import base64
import json
import re
import time
from pathlib import Path

import anthropic

DATA = Path("../SROIE2019")  # dataset lives in the sibling tesseract repo, not duplicated here
FIELDS = ["company", "date", "address", "total"]

PROMPT = """This is a scanned retail receipt. Read it and return ONLY a JSON \
object with these four fields, each a string copied exactly as printed \
(no reformatting dates, no adding currency symbols that aren't there):

{"company": "...", "date": "...", "address": "...", "total": "..."}

If a field is not visible on the receipt, use an empty string. Return only \
the JSON object, nothing else."""


# ---------- VLM call (cached) ----------

def call_vlm(client, model, img_path, out_file, retries=3):
    if out_file.exists():
        return json.loads(out_file.read_text(encoding="utf-8"))

    media_type = "image/jpeg" if img_path.suffix.lower() in (".jpg", ".jpeg") else "image/png"
    image_b64 = base64.standard_b64encode(img_path.read_bytes()).decode()

    for attempt in range(retries):
        try:
            resp = client.messages.create(
                model=model,
                max_tokens=512,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": image_b64}},
                        {"type": "text", "text": PROMPT},
                    ],
                }],
            )
            raw = resp.content[0].text
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            fields = json.loads(match.group(0)) if match else {}
            break
        except Exception as e:
            if attempt == retries - 1:
                fields = {"error": str(e)}
            else:
                time.sleep(2 ** attempt)

    result = {k: fields.get(k, "") for k in FIELDS}
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


# ---------- scoring ----------

def normalize(s):
    return re.sub(r"\s+", " ", s or "").strip().upper()


def score(truth, pred):
    return {f: int(normalize(truth.get(f, "")) == normalize(pred.get(f, ""))) for f in FIELDS}


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser(description="Evaluate a Claude model on SROIE key-field extraction")
    ap.add_argument("--split", default="test", choices=["train", "test"])
    ap.add_argument("--model", default="claude-haiku-5-5")
    ap.add_argument("--limit", type=int, default=10, help="0 = all images")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env

    images = sorted((DATA / args.split / "img").glob("*.jpg"))
    if args.limit:
        images = images[: args.limit]
    run_dir = Path(args.out) / "vlm" / args.model
    print(f"{len(images)} images, model={args.model}, cache dir={run_dir}")

    rows = []
    for i, img in enumerate(images, 1):
        truth = json.loads((DATA / args.split / "entities" / f"{img.stem}.txt").read_text(encoding="utf-8"))
        pred = call_vlm(client, args.model, img, run_dir / f"{img.stem}.json")
        rows.append({"id": img.stem, **score(truth, pred)})
        if i % 10 == 0 or i == len(images):
            print(f"  {i}/{len(images)}")

    print(f"\nfield accuracy ({len(rows)} images):")
    for f in FIELDS:
        acc = sum(r[f] for r in rows) / len(rows)
        print(f"  {f:<10} {acc:6.1%}")
    overall = sum(r[f] for r in rows for f in FIELDS) / (len(rows) * len(FIELDS))
    print(f"  {'overall':<10} {overall:6.1%}")


if __name__ == "__main__":
    main()
