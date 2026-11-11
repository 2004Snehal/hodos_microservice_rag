"""
demo_probabilistic.py — Exercises the /v2 probabilistic RAPTOR + multi-hop API.

Needs the service running (uvicorn app.main:app --port 8000), Ollama with
qwen3:4b pulled, and optionally Redis for the shared semantic cache.
"""

import json

import requests

BASE_URL = "http://localhost:8000"

POLICY = """
Meal expenses in Mumbai are limited to 1500 rupees per day, inclusive of taxes.
Grade L3 employees follow the standard meal cap unless traveling
internationally, see section 9.3 for exceptions to this rule.

Lodging expenses for Grade L3 employees in metro cities are capped at
6000 rupees per night. Vendor approval is required for all lodging bookings
above the standard corporate rate.

Transport claims must be accompanied by an itemized receipt for
reimbursement. Ride-share transport is capped at Rs. 800 per trip within
city limits.

International travel meal caps are waived entirely for approved business
trips exceeding five days.
"""


def ask(question, **scope):
    body = {"question": question, "doc_id": "travel-policy-2026", **scope}
    r = requests.post(f"{BASE_URL}/v2/ask", json=body, timeout=300).json()
    print(f"\nQ: {question}")
    print(f"   source={r['source']} {r.get('cache_layer') or ''} status={r.get('status')} "
          f"confidence={r.get('confidence')}")
    print(f"   rule={r['rule_type']} limit={r['limit_amount']} period={r.get('period')}")
    print(f"   answer: {r['answer']}")
    if r["source"] == "fresh":
        hops = r["retrieval"]["hops"]
        print(f"   hops: {len(hops)} ({r['retrieval']['stopped_reason']}), "
              f"P(sufficient) per hop: {[h['p_sufficient'] for h in hops]}")
    if r.get("warnings"):
        print(f"   warnings: {r['warnings']}")


def main():
    print("=== health ===")
    print(json.dumps(requests.get(f"{BASE_URL}/v2/health").json(), indent=2))

    print("\n=== 1. ingest ===")
    info = requests.post(f"{BASE_URL}/v2/policy",
                         json={"doc_id": "travel-policy-2026", "text": POLICY}, timeout=600).json()
    print({k: info[k] for k in ("leaf_chunks", "nodes_per_level", "nodes_with_multiple_parents", "summaries")})

    print("\n=== 2. fresh questions ===")
    ask("what is the meal cap for L3 employees in Mumbai?", category="meal", grade="L3", location="Mumbai")
    ask("what is the lodging cap for L3 in metro cities?", category="lodging", grade="L3", location="metro")
    ask("what is the cap on laptop purchases?", category="laptop")

    print("\n=== 3. cache: exact repeat, then a paraphrase ===")
    ask("what is the meal cap for L3 employees in Mumbai?", category="meal", grade="L3", location="Mumbai")
    ask("How much can an L3 employee spend on food per day in Mumbai?", category="meal", grade="L3", location="Mumbai")


if __name__ == "__main__":
    main()
