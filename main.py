"""
main.py — Entry point for the Customer Churn Retention Agent System.
"""
from __future__ import annotations

import logging
import sys

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> int:
    from pipeline import run_full_pipeline
    logger.info("Starting Customer Churn Retention Agent Pipeline")
    results = run_full_pipeline()

    success_count = sum(1 for r in results if r.get("status") == "success")
    total = len(results)
    logger.info("Pipeline complete: %d/%d customers processed successfully", success_count, total)

    print("\n" + "=" * 60)
    print("PIPELINE RESULTS")
    print("=" * 60)
    for r in results:
        if r.get("status") == "success":
            passed = "PASSED" if r.get("passed") else "FAILED"
            cid = r["customer_id"]
            iters = r["iterations"]
            html = r.get("html_path", "")
            print(f"  {cid}: {passed} ({iters} eval iteration(s))")
            print(f"    Report: {html}")
        else:
            print(f"  {r['customer_id']}: ERROR — {r.get('error', 'unknown error')}")

    return 0 if success_count == total else 1


if __name__ == "__main__":
    sys.exit(main())
