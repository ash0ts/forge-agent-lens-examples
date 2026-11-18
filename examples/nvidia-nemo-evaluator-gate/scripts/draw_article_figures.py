"""Generate the two bespoke SVG article figures from saved run evidence.

Usage:
    uv run python scripts/draw_article_figures.py --output out/figures
"""
from __future__ import annotations

import argparse
import html
import json
from math import atan2, cos, sin
from pathlib import Path

EXAMPLES_ROOT = Path(__file__).resolve().parents[2]

COLORS = {
    "paper": "#F5F5F2",
    "white": "#FFFFFF",
    "panel": "#EAEAE4",
    "line": "#CFCFC7",
    "ink": "#181816",
    "muted": "#66665E",
    "accent": "#E85D04",
    "candidate_a": "#0B6E99",
}
C = COLORS


class Illustration:
    def __init__(self, title: str, *, width: int = 1200, height: int = 675):
        self.width, self.height = width, height
        self.items = [
            f"<title>{html.escape(title)}</title>",
            f'<rect width="{width}" height="{height}" fill="{C["paper"]}"/>',
        ]

    def rect(self, x, y, w, h, *, fill="none", stroke=None, sw=1.6, rx=0):
        self.items.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" '
            f'stroke="{stroke or C["ink"]}" stroke-width="{sw}"/>'
        )

    def path(self, d, *, fill="none", stroke=None, sw=1.6, dash=None):
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        self.items.append(
            f'<path d="{d}" fill="{fill}" stroke="{stroke or C["ink"]}" stroke-width="{sw}" '
            f'stroke-linejoin="round" stroke-linecap="round"{dash_attr}/>'
        )

    def line(self, x1, y1, x2, y2, **kwargs):
        self.path(f"M{x1} {y1}L{x2} {y2}", **kwargs)

    def text(self, text, x, y, *, size=18, width=400, height=None, weight=400, mono=False, fill=None):
        height = height or size * 1.5
        fam = "IBM Plex Mono" if mono else "IBM Plex Sans"
        self.items.append(
            f'<text x="{x}" y="{y + size}" font-family="{fam}" font-size="{size}" '
            f'font-weight="{weight}" fill="{fill or C["ink"]}" data-box="{x} {y} {width} {height}">'
            f"{html.escape(str(text))}</text>"
        )

    def circle(self, x, y, r, *, fill="none", stroke=None, sw=1.6):
        self.items.append(
            f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}" stroke="{stroke or C["ink"]}" stroke-width="{sw}"/>'
        )

    def arrow(self, x1, y1, x2, y2, *, color=None, dashed=False):
        a = atan2(y2 - y1, x2 - x1)
        self.line(x1, y1, x2, y2, stroke=color, dash="6 5" if dashed else None)
        pts = [
            (x2 - 10 * cos(a) + 5 * sin(a), y2 - 10 * sin(a) - 5 * cos(a)),
            (x2, y2),
            (x2 - 10 * cos(a) - 5 * sin(a), y2 - 10 * sin(a) + 5 * cos(a)),
        ]
        self.path("M" + "L".join(f"{px:.2f} {py:.2f}" for px, py in pts), stroke=color)

    def label(self, number, text, x, y, tx, ty, *, width=280):
        self.path(f"M{x + 14} {y + 26}V{ty}H{tx}", stroke=C["muted"])
        self.circle(tx, ty, 3, fill=C["ink"], sw=0)
        self.circle(x + 14, y + 13, 13, fill=C["accent"], sw=0)
        self.text(number, x + 8, y + 1, size=17, width=20, weight=600)
        self.text(text, x + 36, y - 1, width=width, size=18, weight=600)

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        content = (
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.width} {self.height}" '
            f'font-family="IBM Plex Sans" fill="{C["ink"]}">' + "".join(self.items) + "</svg>"
        )
        path.write_text(content, encoding="utf-8")
        return path


def gate_figure(out_path: Path) -> Path:
    d = Illustration("NeMo Evaluator release gate on Forge", width=1200, height=680)
    d.text("RELEASE GATE / LIVE 2026-11-18 (ENABLE_THINKING = FALSE)", 36, 14, size=14, mono=True, width=760, fill=C["muted"])
    d.text("One ticket through three prompts, all 48 held-out cells, and the nine policy checks.", 36, 38, size=24, weight=600, width=1128)
    d.line(36, 82, 1164, 82, stroke=C["line"])

    d.text("01  One held-out ticket", 36, 96, size=19, weight=600, width=340)
    d.rect(36, 132, 344, 86, fill=C["panel"], stroke=C["ink"])
    d.text('v2-inference-09  ·  expected: "inference"', 48, 142, size=13, mono=True, width=320, fill=C["muted"])
    d.text('"API key is valid and balance is positive;', 48, 166, size=14, width=320)
    d.text('calls to deployed model get permission denied."', 48, 186, size=14, width=325)

    prompts = [
        ("baseline", "one-line prompt", '"training"', "42 / 48", C["ink"], False),
        ("candidate", "defines the 4 queues", '"inference"', "48 / 48", C["candidate_a"], True),
        ("control", 'always returns "billing"', '"billing"', "2 / 8", C["accent"], False),
    ]
    for i, (name, desc, out_label, ratio, col, ok) in enumerate(prompts):
        y = 236 + i * 82
        d.line(208, 218 if i == 0 else y - 14, 208, y, stroke=C["line"])
        d.rect(36, y, 344, 68, fill=C["white"], stroke=col, sw=2.0 if ok else 1.5)
        d.text(name, 48, y + 8, size=17, weight=600, width=130)
        d.text(desc, 48, y + 34, size=13.5, width=175, fill=C["muted"])
        d.rect(224, y + 10, 144, 26, fill=C["panel"] if ok else C["paper"], stroke=col, sw=1.2, rx=4)
        mark = "ok" if ok else "miss"
        d.text(f"{out_label} ({mark})", 232, y + 14, size=12.5, mono=True, width=132, fill=col if not ok else C["ink"])
        d.text(f"held-out: {ratio}", 232, y + 42, size=13, mono=True, width=140, fill=C["muted"])
        d.arrow(382, y + 34, 416, y + 34, color=col)

    d.text("02  Score with NeMo Evaluator", 424, 96, size=19, weight=600, width=340)
    d.rect(424, 132, 336, 336, fill=C["panel"], stroke=C["ink"])
    d.text("nemo_evaluator.exact_match", 438, 142, size=14, mono=True, width=308)
    d.text("48 held-out tickets (12 per queue)", 438, 162, size=13.5, width=308, fill=C["muted"])

    cats = [("billing", 0), ("inference", 3), ("observability", 1), ("training", 2)]
    for ci, (cat, fixed_n) in enumerate(cats):
        y = 190 + ci * 50
        base_ok = 12 - fixed_n
        d.text(f"{cat} ({base_ok}→12)", 438, y, size=12.5, mono=True, width=180)
        for k in range(12):
            cx = 438 + k * 25
            cy = y + 20
            if k < fixed_n:
                d.rect(cx, cy, 18, 18, fill=C["accent"], stroke=C["ink"], sw=1.2)
            else:
                d.rect(cx, cy, 18, 18, fill=C["ink"], stroke="none")

    d.line(438, 396, 744, 396, stroke=C["line"])
    d.text("control (8 tickets: 2 billing hits, 6 misses)", 438, 404, size=12.5, mono=True, width=306, fill=C["muted"])
    for k in range(8):
        cx = 438 + k * 25
        d.rect(cx, 426, 18, 18, fill=C["ink"] if k < 2 else C["white"], stroke=C["accent"] if k >= 2 else C["ink"], sw=1.4)
    d.text("orange = baseline missed, candidate fixed", 438, 448, size=12.5, width=306, fill=C["muted"])

    d.arrow(762, 300, 790, 300, color=C["ink"])

    d.text("03  Apply policy (gate.py)", 796, 96, size=19, weight=600, width=368)
    checks = [
        ("valid_membership", "128 = 128"),
        ("required_heldout_count", "48 = 48"),
        ("complete_responses", "128 scored"),
        ("strict_label_contract", "0 invalid"),
        ("candidate_absolute_floor", "1.00 >= 0.85"),
        ("every_category_floor", "1.00 >= 0.75"),
        ("candidate_delta_floor", "+0.125 >= 0"),
        ("paired_regression_budget", "0 <= 0"),
        ("negative_control_rejected", "0.25 < 0.85"),
    ]
    for i, (name, rule) in enumerate(checks):
        y = 132 + i * 31
        d.rect(796, y, 368, 25, fill=C["white"], stroke=C["line"])
        d.text(name, 804, y + 4, size=12, mono=True, width=196)
        d.text(rule, 1004, y + 4, size=12, mono=True, width=104, fill=C["muted"])
        d.text("pass", 1118, y + 4, size=12, mono=True, weight=600, width=40, fill=C["candidate_a"])

    d.rect(796, 420, 368, 48, fill=C["white"], stroke=C["candidate_a"], sw=2.2)
    d.text('decision = "pass"  ·  ci_exit_code = 0', 810, 434, size=15, mono=True, weight=600, width=344)

    d.label("1", "6 boundary fixes, 0 regressions", 36, 514, 447, 268, width=330)
    d.label("2", "Control scores 0.25 and is rejected", 424, 514, 500, 435, width=330)
    d.label("3", "CI reads run summary; deploys nothing", 796, 514, 980, 468, width=360)

    d.line(36, 586, 1164, 586, stroke=C["line"])
    d.text('decision = "pass" if all(checks.values()) else "hold"   # evaluator_gate/gate.py', 36, 598, size=15, mono=True, width=1128)
    d.text(
        "Live 2026-11-18 run e0ce3715 (thinking off: 179 ms median latency, 132 output tokens across 48 held-out tickets).",
        36, 630, size=15.5, width=1128, fill=C["muted"],
    )
    return d.save(out_path)


def timeline_figure(out_path: Path) -> Path:
    led_path = EXAMPLES_ROOT / "nvidia-relay-agent-lens/evidence/live-2026-11-18/span-ledger.json"
    led = json.loads(led_path.read_text())
    d = Illustration("Twenty Relay spans exported over OTLP to Agent Lens", width=1200, height=670)
    d.text("NEMO RELAY → AGENT LENS / LIVE OTLP / 20 SPANS ACROSS 56.9 SECONDS", 36, 14, size=14, mono=True, width=820, fill=C["muted"])
    d.text("One tool call fails on the third task. The next model call reads the error and retries.", 36, 38, size=24, weight=600, width=1128)
    d.line(36, 82, 1164, 82, stroke=C["line"])

    agents = [s for s in led if s["operation"] == "invoke_agent"]
    x0, x1 = 304, 1154
    total = 57000.0

    def X(ms: float) -> float:
        return x0 + (ms / total) * (x1 - x0)

    for sec in range(0, 60, 10):
        x = X(sec * 1000)
        d.line(x, 112, x, 120, stroke=C["muted"])
        d.text(f"{sec}s", x - 10, 92, size=13, mono=True, width=40, fill=C["muted"])
    d.line(x0, 120, x1, 120, stroke=C["line"])

    lane_ys = [142, 260, 378]
    for ai, a in enumerate(agents):
        y = lane_ys[ai]
        kids = [
            s for s in led
            if s["parent_span_id"] == a["span_id"]
            or (a["offset_ms"] <= s["offset_ms"] < a["offset_ms"] + a["duration_ms"] and s["operation"] != "invoke_agent")
        ]
        seen = set()
        ordered = []
        for s in sorted(kids, key=lambda item: item["offset_ms"]):
            if s["span_id"] not in seen:
                seen.add(s["span_id"])
                ordered.append(s)
        n_chats = sum(s["operation"] == "chat" for s in ordered)
        dur_s = a["duration_ms"] / 1000.0
        d.text(f"Turn {ai + 1} · {a['task_id']}", 36, y + 2, size=15.5, weight=600, width=258)
        d.text(f"invoke_agent · {dur_s:.1f}s · {n_chats} chats", 36, y + 26, size=12.5, mono=True, width=258, fill=C["muted"])

        d.rect(X(a["offset_ms"]), y, max(2, X(a["offset_ms"] + a["duration_ms"]) - X(a["offset_ms"])), 18, fill=C["panel"], stroke=C["ink"])

        for s in ordered:
            w = max(3, X(s["offset_ms"] + s["duration_ms"]) - X(s["offset_ms"]))
            if s["operation"] == "chat":
                d.rect(X(s["offset_ms"]), y + 26, max(2, w - 1.5), 16, fill=C["candidate_a"], stroke="none")
            else:
                is_err = s["status_code"] == 2
                tx = X(s["offset_ms"])
                d.rect(tx - 2.5, y + 48, 6, 18, fill=C["accent"] if is_err else C["ink"], stroke="none")
                short = "schema" if "schema" in s["name"] else "query"
                if ai < 2:
                    d.text(short, tx - 14, y + 70, size=11.5, mono=True, width=54, fill=C["muted"])
                else:
                    if "schema" in s["name"]:
                        d.text("schema", tx - 22, y + 70, size=11.5, mono=True, width=56, fill=C["muted"])
                    elif is_err:
                        d.path(f"M{tx:.1f} {y + 68}V{y + 99}H{tx - 16:.1f}", stroke=C["accent"], sw=1.6)
                        d.text(
                            "query → ERROR (TimeoutError)",
                            tx - 252, y + 89, size=13, mono=True, weight=600, width=240, fill=C["accent"],
                        )
                    else:
                        d.path(f"M{tx:.1f} {y + 68}V{y + 123}H{tx - 12:.1f}", stroke=C["ink"], sw=1.4)
                        d.text(
                            "retry query → ok ([[1106, 1244]])",
                            tx - 262, y + 113, size=13, mono=True, width=254, fill=C["ink"],
                        )

    ly = 548
    d.rect(36, ly, 26, 14, fill=C["panel"], stroke=C["ink"])
    d.text("invoke_agent (1 turn)", 70, ly - 3, size=14, width=185)
    d.rect(270, ly, 26, 14, fill=C["candidate_a"], stroke="none")
    d.text("chat (model + thinking)", 304, ly - 3, size=14, width=210)
    d.rect(534, ly - 1, 6, 16, fill=C["ink"], stroke="none")
    d.text("execute_tool ok (<1 ms)", 550, ly - 3, size=14, width=215)
    d.rect(784, ly - 1, 6, 16, fill=C["accent"], stroke="none")
    d.text("execute_tool status ERROR", 800, ly - 3, size=14, width=240)

    d.line(36, 584, 1164, 584, stroke=C["line"])
    d.text(
        "Offsets and durations from relay-genai.jsonl. Model bars include Nemotron 3.5 Lightning thinking tokens (7,177 output tokens total).",
        36, 598, size=15, width=1128, fill=C["muted"],
    )
    return d.save(out_path)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=Path("out/figures"))
    args = p.parse_args()
    print(gate_figure(args.output / "gate-device.svg"))
    print(timeline_figure(args.output / "span-timeline.svg"))
