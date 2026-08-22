"""
demo.py — exercises both API endpoints against a running instance of the service.

Usage:
    # in one terminal:
    uvicorn app.main:app --reload

    # in another terminal:
    python scripts/demo.py
"""

import requests

BASE_URL = "http://localhost:8000"

SAMPLE_POLICY = """
Meal expenses in Mumbai are limited to 1500 rupees per day, inclusive of taxes.
Grade L3 employees follow the standard meal cap unless traveling internationally,
see section 9.3 for exceptions to this rule.

Lodging expenses for Grade L3 employees in metro cities are capped at 6000 rupees
per night. Vendor approval is required for all lodging bookings above the
standard corporate rate.

Transport claims must be accompanied by an itemized receipt for reimbursement.
Ride-share transport is capped at 800 rupees per trip within city limits.

International travel meal caps are waived entirely for approved business trips
exceeding five days, per section 9.3. Precedent review applies to any claim
exceeding twice the median historical claim amount for that category.
"""


def main():
    print("=== 1. Ingesting policy document ===")
    resp = requests.post(
        f"{BASE_URL}/policy",
        json={"doc_id": "travel-policy-2026", "text": SAMPLE_POLICY},
    )
    print(resp.status_code, resp.json())

    print("\n=== 2. Asking a question ===")
    resp = requests.post(
        f"{BASE_URL}/ask",
        json={"question": "What is the meal cap for L3 employees in Mumbai?", "top_k": 3},
    )
    print(resp.status_code)
    data = resp.json()
    print("Answer:\n", data["answer"])
    print("\nCitations:")
    for c in data["citations"]:
        print(f"  - [{c['chunk_id']}] score={c['score']}: {c['text'][:100]}...")

    print("\n=== 3. Health check ===")
    resp = requests.get(f"{BASE_URL}/health")
    print(resp.status_code, resp.json())


if __name__ == "__main__":
    main()
