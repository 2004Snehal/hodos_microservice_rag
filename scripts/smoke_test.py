"""
smoke_test.py — End-to-end checks for every endpoint of a running service.

  python scripts/smoke_test.py                          # http://localhost:8000
  python scripts/smoke_test.py --base-url http://host:8000
  python scripts/smoke_test.py --skip-llm               # only the deterministic /v1 endpoints

Generates its own bill fixtures (PDF with a text layer, printed PNG, and —
when macOS handwriting fonts are available — a handwritten JPG and an
image-only PDF). Each run ingests policy text containing a fresh nonce, so
answers cached in Redis by earlier runs can't make a fresh check pass.
Exits non-zero if any check fails.
"""

import argparse
import sys
import tempfile
import time
import uuid
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

POLICY_V1 = """Meal expenses in Mumbai are limited to Rs. 1500 per day, inclusive of taxes.

Lodging expenses for Grade L3 employees in metro cities are capped at Rs. 6000 per night.

Ride-share transport is capped at Rs. 800 per trip within city limits."""

POLICY_V2 = """Meal expenses in Mumbai are limited to 1500 rupees per day, inclusive of taxes.
Grade L3 employees follow the standard meal cap unless traveling
internationally, see section 9.3 for exceptions to this rule.

Lodging expenses for Grade L3 employees in metro cities are capped at
6000 rupees per night. Vendor approval is required for all lodging bookings
above the standard corporate rate.

Transport claims must be accompanied by an itemized receipt for
reimbursement. Ride-share transport is capped at Rs. 800 per trip within
city limits.

International travel meal caps are waived entirely for approved business
trips exceeding five days."""


class Checker:
    def __init__(self):
        self.results = []

    def check(self, name, condition, detail=""):
        self.results.append((name, bool(condition), detail))
        mark = "PASS" if condition else "FAIL"
        print(f"  [{mark}] {name}" + (f" — {detail}" if detail and not condition else ""))
        return condition

    def summary(self):
        failed = [r for r in self.results if not r[1]]
        print(f"\n{len(self.results) - len(failed)}/{len(self.results)} checks passed")
        for name, _, detail in failed:
            print(f"  FAILED: {name} — {detail}")
        return not failed


def make_fixtures(directory: Path):
    import pymupdf

    bills = {}
    text = ("Hotel Sea View, Mumbai\nDate: 14/09/2026\nHotel room stay 2 nights Rs. 11,000\n"
            "Dinner at hotel restaurant Rs. 1,400\nLunch Rs. 1,900\nTaxi to airport Rs. 950\n"
            "Uber within city Rs. 420")
    doc = pymupdf.open()
    doc.new_page().insert_text((50, 72), text, fontsize=12)
    doc.save(directory / "digital.pdf")
    doc[0].get_pixmap(dpi=200).save(directory / "printed.png")
    bills["digital_pdf"] = directory / "digital.pdf"
    bills["printed_png"] = directory / "printed.png"

    try:
        from app.core.bill_router.synth import render_bill
        image = render_bill("handwritten", "Chalkduster.ttf", 998)
        image.save(directory / "handwritten.jpg")
        bills["handwritten_jpg"] = directory / "handwritten.jpg"
        scan = pymupdf.open()
        page = scan.new_page(width=image.width * 0.6, height=image.height * 0.6)
        page.insert_image(page.rect, filename=str(directory / "handwritten.jpg"))
        scan.save(directory / "handwritten_scan.pdf")
        bills["handwritten_scan_pdf"] = directory / "handwritten_scan.pdf"
    except (FileNotFoundError, ImportError) as exc:
        print(f"  (handwritten fixtures skipped: {exc})")
    return bills


def post_file(base, endpoint, path, **form):
    with open(path, "rb") as f:
        return requests.post(f"{base}{endpoint}", files={"file": (path.name, f)}, data=form, timeout=600)


def test_v1(base, c: Checker, bills):
    print("\n== /v1: /health /policy /ask /audit")
    r = requests.get(f"{base}/health", timeout=10)
    c.check("GET /health 200", r.status_code == 200, r.text[:200])

    r = requests.post(f"{base}/policy", json={"doc_id": "smoke-v1", "text": "  "}, timeout=30)
    c.check("POST /policy empty -> 400", r.status_code == 400, r.text[:200])
    r = requests.post(f"{base}/policy", json={"doc_id": "smoke-v1", "text": POLICY_V1}, timeout=60)
    c.check("POST /policy ingests", r.status_code == 200 and r.json()["chunks_created"] >= 1, r.text[:200])

    r = requests.post(f"{base}/ask", json={"question": " "}, timeout=30)
    c.check("POST /ask empty -> 400", r.status_code == 400, r.text[:200])
    r = requests.post(f"{base}/ask", json={"question": "What is the meal cap in Mumbai?", "top_k": 2}, timeout=60)
    body = r.json() if r.ok else {}
    c.check("POST /ask returns cited passage with 1500",
            r.ok and body.get("citations") and "1500" in body.get("answer", ""), r.text[:300])

    r = post_file(base, "/audit", Path(__file__))
    c.check("POST /audit unsupported type -> 400", r.status_code == 400, r.text[:200])

    r = post_file(base, "/audit", bills["digital_pdf"])
    body = r.json() if r.ok else {}
    routes = body.get("page_routes", [])
    verdicts = {v["category"]: v for v in body.get("category_verdicts", [])}
    c.check("/audit digital PDF routed to direct_text",
            routes and routes[0]["route"] == "digital" and routes[0]["method"] == "direct_text", str(routes))
    c.check("/audit parses taxi as transport (not lodging)", "transport" in verdicts, str(list(verdicts)))
    c.check("/audit meal 3300 vs cap 1500 -> fail",
            verdicts.get("meal", {}).get("verdict") == "fail", str(verdicts.get("meal")))

    r = post_file(base, "/audit", bills["printed_png"])
    routes = r.json().get("page_routes", []) if r.ok else []
    c.check("/audit printed PNG routed printed -> ocr",
            routes and routes[0]["route"] == "printed" and routes[0]["method"] == "ocr", str(routes))

    for key in ("handwritten_jpg", "handwritten_scan_pdf"):
        if key not in bills:
            continue
        r = post_file(base, "/audit", bills[key])
        routes = r.json().get("page_routes", []) if r.ok else []
        ok = routes and routes[0]["route"] == "handwritten" and (
            routes[0]["method"] == "vlm" or any("no VLM" in n for n in routes[0]["notes"]))
        c.check(f"/audit {key} routed handwritten -> vlm (or OCR + no-VLM note)", ok, str(routes))


def test_v2(base, c: Checker, bills):
    print("\n== /v2: health, policy, ask, cache, audit")
    health = requests.get(f"{base}/v2/health", timeout=30).json()
    c.check("GET /v2/health: LLM available", health["llm"]["available"], str(health["llm"]))
    c.check("GET /v2/health: bill router trained", health["bill_router"]["trained"], str(health["bill_router"]))
    print(f"  (cache backend: {health['cache']['backend']}, VLM available: {health['vlm']['available']})")

    nonce = uuid.uuid4().hex[:8]
    doc_id = "smoke-v2"
    text = f"{POLICY_V2}\n\nPolicy reference {nonce}."
    t = time.time()
    r = requests.post(f"{base}/v2/policy", json={"doc_id": doc_id, "text": text}, timeout=900)
    info = r.json() if r.ok else {}
    c.check("POST /v2/policy builds a multi-level tree",
            r.ok and len(info.get("nodes_per_level", {})) >= 2, r.text[:300])
    print(f"  (ingest {time.time() - t:.1f}s, levels {info.get('nodes_per_level')}, summaries {info.get('summaries')})")

    r = requests.post(f"{base}/v2/ask", json={"question": "x", "doc_id": "no-such-doc"}, timeout=30)
    c.check("POST /v2/ask unknown doc -> 404", r.status_code == 404, r.text[:200])
    r = requests.post(f"{base}/v2/ask", json={"question": " ", "doc_id": doc_id}, timeout=30)
    c.check("POST /v2/ask empty -> 400", r.status_code == 400, r.text[:200])

    def ask(question, **scope):
        t0 = time.time()
        resp = requests.post(f"{base}/v2/ask", json={"question": question, "doc_id": doc_id, **scope}, timeout=600)
        return (resp.json() if resp.ok else {"error": resp.text}), time.time() - t0

    meal_scope = dict(category="meal", grade="L3", location="Mumbai")
    cases = [
        ("what is the meal cap for L3 employees in Mumbai?", meal_scope, "cap", 1500, "ok"),
        ("what is the lodging cap for L3 in metro cities?", dict(category="lodging", grade="L3", location="metro"), "cap", 6000, "ok"),
        ("how much can I claim for an Uber ride within the city?", dict(category="transport"), "cap", 800, "ok"),
        ("what is the cap on laptop purchases?", dict(category="laptop"), "no_rule_found", None, "ok"),
    ]
    for question, scope, rule, limit, status in cases:
        body, dt = ask(question, **scope)
        got_limit = body.get("limit_amount")
        ok = (body.get("rule_type") == rule and body.get("status") == status
              and (limit is None or (got_limit is not None and float(got_limit) == limit)))
        c.check(f"/v2/ask '{question[:45]}' -> {rule} {limit or ''}", ok,
                f"got {body.get('rule_type')} {got_limit} status={body.get('status')} conf={body.get('confidence')}")
        print(f"     ({dt:.1f}s, confidence {body.get('confidence')}, source {body.get('source')})")

    body, dt = ask("what is the meal cap for L3 employees in Mumbai?", **meal_scope)
    c.check("/v2/ask exact repeat served from cache (L1/L2)",
            body.get("source") == "cache" and body.get("cache_layer") in ("l1_exact", "l2_exact"), str(body.get("cache_layer")))
    body, dt = ask("How much can an L3 employee spend on food per day in Mumbai?", **meal_scope)
    c.check("/v2/ask paraphrase served from semantic cache (L3)",
            body.get("cache_layer") == "l3_semantic", f"source={body.get('source')} layer={body.get('cache_layer')}")
    body, dt = ask("is the meal cap waived?", **meal_scope)
    c.check("/v2/ask different question in same scope NOT served from cache", body.get("source") == "fresh",
            f"source={body.get('source')} layer={body.get('cache_layer')} sim={body.get('cache_similarity')}")

    r = post_file(base, "/v2/audit", bills["digital_pdf"], doc_id="no-such-doc")
    c.check("POST /v2/audit unknown doc -> 404", r.status_code == 404, r.text[:200])
    t = time.time()
    r = post_file(base, "/v2/audit", bills["digital_pdf"], doc_id=doc_id, grade="L3", location="Mumbai", days=2)
    body = r.json() if r.ok else {}
    verdicts = {v["category"]: v["verdict"] for v in body.get("category_verdicts", [])}
    c.check("/v2/audit 2 days: lodging 11000 <= 2x6000 -> pass", verdicts.get("lodging") == "pass", str(verdicts))
    c.check("/v2/audit 2 days: meal 3300 > 2x1500 -> fail", verdicts.get("meal") == "fail", str(verdicts))
    c.check("/v2/audit transport has a verdict", "transport" in verdicts, str(verdicts))
    c.check("/v2/audit overall fail", body.get("overall") == "fail", str(body.get("overall")))
    print(f"     ({time.time() - t:.1f}s, verdicts {verdicts})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--skip-llm", action="store_true")
    args = parser.parse_args()
    base = args.base_url.rstrip("/")

    c = Checker()
    with tempfile.TemporaryDirectory() as tmp:
        bills = make_fixtures(Path(tmp))
        test_v1(base, c, bills)
        if not args.skip_llm:
            test_v2(base, c, bills)
    sys.exit(0 if c.summary() else 1)


if __name__ == "__main__":
    main()
