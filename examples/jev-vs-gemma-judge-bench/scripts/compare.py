"""Per-field agreement of Jev vs Gemma on the same turns, plus latency and token cost."""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from jev_port import failure_categories, fired

GEMMA_INPUT_PRICE = 0.10 / 1e6
GEMMA_OUTPUT_PRICE = 0.34 / 1e6
# $42 per billion input tokens; output tokens confirmed free.
JEV_INPUT_PRICE = 42 / 1e9
JEV_OUTPUT_PRICE = 0.0


def load(path: Path) -> dict[tuple[int, str], dict]:
    rows = {}
    for line in path.read_text().splitlines():
        record = json.loads(line)
        # Re-derive label sets from the stored probabilities so threshold changes need no rerun.
        if "answers" in record and record["judge"] == "intent":
            record["outcome"]["items"] = fired(record["answers"])
        if "answers" in record and record["judge"] == "failure":
            record["outcome"]["items"] = failure_categories(record["answers"])
        rows[(record["turn"], record["judge"])] = record

    return rows


def set_metrics(reference: list[str], candidate: list[str]) -> tuple[int, int, int]:
    ref, cand = set(reference), set(candidate)
    return len(ref & cand), len(cand - ref), len(ref - cand)


def main(run_dir: Path) -> None:
    gemma = load(run_dir / "gemma.jsonl")
    jev = load(run_dir / "jev.jsonl")
    turns = sorted({turn for turn, _ in jev})

    fields = {"status": 0, "primary_in_gemma": 0, "sentiment": 0, "sentiment_target": 0, "language": 0}
    intent_tp = intent_fp = intent_fn = 0
    fail_agree = 0
    fail_tp = fail_fp = fail_fn = 0
    cat_tp = cat_fp = cat_fn = 0
    gemma_parse_fail = 0
    disagreements = []
    for turn in turns:
        g_i, j_i = gemma[(turn, "intent")], jev[(turn, "intent")]
        g_f, j_f = gemma[(turn, "failure")], jev[(turn, "failure")]
        if g_i["outcome"] is None or g_f["outcome"] is None:
            gemma_parse_fail += 1
            continue
        gi, ji = g_i["outcome"], j_i["outcome"]
        gemma_content = json.loads(g_i["content"])
        gemma_language = gemma_content.get("language", "?")
        fields["status"] += gi["status"] == ji["status"]
        fields["primary_in_gemma"] += ji["primary_category"] in gi["items"]
        fields["sentiment"] += gi["sentiment"] == ji["sentiment"]
        fields["sentiment_target"] += gi["sentiment_target"] == ji["sentiment_target"]
        fields["language"] += gemma_language == ji["language"]
        tp, fp, fn = set_metrics(gi["items"], ji["items"])
        intent_tp, intent_fp, intent_fn = intent_tp + tp, intent_fp + fp, intent_fn + fn

        gf, jf = g_f["outcome"], j_f["outcome"]
        g_failed = gf["status"] == "failed"
        j_failed = jf["failed_p"] >= 0.5 or jf["tool_error_p"] >= 0.5
        fail_agree += g_failed == j_failed
        fail_tp += g_failed and j_failed
        fail_fp += (not g_failed) and j_failed
        fail_fn += g_failed and not j_failed
        g_cats = sorted({item.split(":")[0] for item in gf["items"]})
        tp, fp, fn = set_metrics(g_cats, jf["items"])
        cat_tp, cat_fp, cat_fn = cat_tp + tp, cat_fp + fp, cat_fn + fn
        if gi["status"] != ji["status"] or set(gi["items"]) != set(ji["items"]) or g_failed != j_failed or g_cats != jf["items"] or gi["sentiment"] != ji["sentiment"]:
            disagreements.append((turn, gi, ji, gf["items"], jf["items"], f"{jf['failed_p']:.2f}/{jf['tool_error_p']:.2f} errsteps={j_f['tool_error_steps']}"))

    n = len(turns) - gemma_parse_fail
    print(f"turns={len(turns)} gemma_parse_failures={gemma_parse_fail} scored={n}")
    for name, hits in fields.items():
        print(f"  intent.{name:<18} agree {hits}/{n} = {hits / n:.0%}")
    print(f"  intent.category_set     vs gemma: P={_ratio(intent_tp, intent_tp + intent_fp)} R={_ratio(intent_tp, intent_tp + intent_fn)} (tp={intent_tp} fp={intent_fp} fn={intent_fn})")
    print(f"  failure.fired           agree {fail_agree}/{n} = {fail_agree / n:.0%} (tp={fail_tp} fp={fail_fp} fn={fail_fn}); gemma fired {fail_tp + fail_fn}, jev fired {fail_tp + fail_fp}")
    print(f"  failure.category_set    vs gemma: P={_ratio(cat_tp, cat_tp + cat_fp)} R={_ratio(cat_tp, cat_tp + cat_fn)} (tp={cat_tp} fp={cat_fp} fn={cat_fn})")

    for judge in ("intent", "failure"):
        j_lat = [jev[(t, judge)]["latency_s"] for t in turns]
        j_in = sum(jev[(t, judge)]["usage"]["input_tokens"] for t in turns)
        j_out = sum(jev[(t, judge)]["usage"]["output_tokens"] for t in turns)
        g_in = sum(gemma[(t, judge)]["prompt_tokens"] for t in turns)
        g_out = sum(gemma[(t, judge)]["completion_tokens"] for t in turns)
        g_cost = g_in * GEMMA_INPUT_PRICE + g_out * GEMMA_OUTPUT_PRICE
        j_cost = j_in * JEV_INPUT_PRICE + j_out * JEV_OUTPUT_PRICE
        print(f"  {judge}: jev latency p50 {statistics.median(j_lat):.2f}s p95 {sorted(j_lat)[int(0.95 * (len(j_lat) - 1))]:.2f}s max {max(j_lat):.2f}s")
        print(f"  {judge}: tokens/turn jev in={j_in / len(turns):.0f} out={j_out / len(turns):.0f} (${j_cost / len(turns) * 1000:.3f}/1k turns) | "
              f"gemma in={g_in / len(turns):.0f} out={g_out / len(turns):.0f} (${g_cost / len(turns) * 1000:.3f}/1k turns)")

    print("\ndisagreements (turn | gemma intent | jev intent | gemma failures | jev failures | jev failed_p):")
    for row in disagreements:
        turn, gi, ji, gf_items, jf_items, p = row
        print(f"  t{turn}: G {gi['status']} {gi['items']} {gi['sentiment']}/{gi['sentiment_target']} | J {ji['status']} {ji['items']} {ji['sentiment']}/{ji['sentiment_target']} | GF {gf_items} | JF {jf_items} p={p}")


def _ratio(num: int, den: int) -> str:
    return "n/a" if den == 0 else f"{num / den:.0%}"


if __name__ == "__main__":
    main(Path(sys.argv[1]))
