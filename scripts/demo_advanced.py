"""
demo_advanced.py — Exercises the integrated pipeline (app/advanced/integrated/pipeline.py).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.advanced.integrated.pipeline import AdvancedRagPipeline
from app.core.bill_parser import CATEGORY_KEYWORDS

SAMPLE_POLICY = """
Meal expenses in Mumbai are limited to Rs. 1500 per day, inclusive of taxes.
Grade L3 employees follow the standard meal cap unless traveling
internationally, see section 9.3 for exceptions to this rule.

Lodging expenses for Grade L3 employees in metro cities are capped at
Rs. 6000 per night. Vendor approval is required for all lodging bookings
above the standard corporate rate.

Transport claims must be accompanied by an itemized receipt for
reimbursement. Ride-share transport is capped at Rs. 800 per trip within
city limits.

International travel meal caps are waived entirely for approved business
trips exceeding five days.
"""


def main():
    cache_dir = Path("/tmp/hodos_advanced_demo_cache")
    pipeline = AdvancedRagPipeline(cache_dir=cache_dir)
    pipeline.cache.clear()  # start from a cold cache so step 2 really runs retrieval
    print(f"cache backend: {pipeline.cache.stats()['backend']}")

    print("=== 1. Building RAPTOR tree for the policy document ===")
    build_result = pipeline.build_tree_for_document("travel-policy-2026", SAMPLE_POLICY)
    print(build_result)

    print("\n=== 2. First ask — RAPTOR + multi-hop retrieval ===")
    result_1 = pipeline.ask(
        question="what is the meal cap for L3 employees in Mumbai?",
        doc_id="travel-policy-2026",
        category="meal",
        grade="L3",
        location="Mumbai",
        version="v1",
        category_keywords=CATEGORY_KEYWORDS["meal"],
    )
    print(f"source: {result_1['source']}")
    print(f"rule_type: {result_1['rule_type']}")
    print(f"limit_amount: {result_1['limit_amount']}")
    print(f"confidence: {result_1['confidence']}")
    print(f"hops_taken: {result_1['hops_taken']}")
    print(f"final_coverage: {result_1['final_coverage']:.2f}")

    print("\n=== 3. Repeat asks — served from cache ===")
    for label, question in [
        ("same question", "what is the meal cap for L3 employees in Mumbai?"),
        ("paraphrase", "what is the daily food limit for L3 in Mumbai"),
    ]:
        result_2 = pipeline.ask(
            question=question,
            doc_id="travel-policy-2026",
            category="meal",
            grade="L3",
            location="Mumbai",
            version="v1",
            category_keywords=CATEGORY_KEYWORDS["meal"],
        )
        print(f"[{label}] source: {result_2['source']} via {result_2.get('cache_layer')} "
              f"(similarity {result_2.get('cache_similarity')}) -> limit_amount: {result_2['limit_amount']}")

    print("\n=== 4. WAIVED category ===")
    result_3 = pipeline.ask(
        question="are meal caps waived for international travel?",
        doc_id="travel-policy-2026",
        category="meal",
        grade="L3",
        location="International",
        version="v1",
        category_keywords=CATEGORY_KEYWORDS["meal"],
    )
    print(f"source: {result_3['source']}")
    print(f"rule_type: {result_3['rule_type']}")
    print(f"limit_amount: {result_3['limit_amount']}")

    print("\n=== 5. Cache stats ===")
    print(pipeline.cache.stats())


if __name__ == "__main__":
    main()

