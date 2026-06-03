"""
evaluate.py — Marketplace Compliance Checker Evaluation
========================================================
Tests whether the AI correctly identifies known violations
and correctly passes known compliant clauses.

Metrics:
  - Critical issue detection rate  : % of planted critical violations caught
  - High issue detection rate       : % of planted high violations caught
  - False positive rate             : % of compliant clauses wrongly flagged
  - Risk score accuracy             : is the risk level (LOW/MED/HIGH/CRITICAL) correct?
  - Overall precision               : caught / (caught + false_positives)

Usage:
  pip install anthropic python-dotenv langchain-anthropic langchain-community
  pip install langchain-text-splitters langchain-huggingface langchain-chroma
  pip install chromadb sentence-transformers pypdf
  ANTHROPIC_API_KEY=... python evaluate.py
"""

import os
import json
import time
import tempfile
from datetime import datetime
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()

GREEN  = "\033[92m"
BLUE   = "\033[94m"
YELLOW = "\033[93m"
RED    = "\033[91m"
RESET  = "\033[0m"
BOLD   = "\033[1m"

# ── Marketplace rules (same as app.py) ────────────────────────────────
MARKETPLACE_RULES = """
# Rakuten & Mercari Marketplace Seller Compliance Rules

## 1. Payment Terms
- Payment to sellers must be processed within 60 days of confirmed delivery
- Late payment penalties must not exceed 14.6% annual interest per Japanese Commercial Code
- Any fee changes require written notice at least 60 days in advance
- Retroactive fee changes are prohibited

## 2. Liability & Indemnification
- Seller liability cap must not exceed 12 months of total fees paid
- Platform cannot disclaim liability for gross negligence or willful misconduct under Japanese law
- Consequential damages waiver must be mutual (cannot apply only to platform)

## 3. Intellectual Property (IP)
- Seller must retain ownership of their brand and product IP
- Platform license to use seller content must be limited to marketplace operations only
- Sublicensing of seller content to third parties requires explicit written consent

## 4. Contract Term & Termination
- Minimum notice period for contract termination must be 30 days
- Immediate termination only permitted for material breach, fraud, or illegal activity
- Seller data must be returned or deleted within 30 days of contract termination
- Funds held in escrow must be released within 15 business days of termination
- Non-compete clauses cannot restrict seller from using other Japanese platforms

## 5. Data Privacy & Security
- All seller and buyer data must be handled per Japan's APPI
- Data breach notification must occur within 3-5 days per Japan APPI 2022 amendment
- Platform cannot sell seller business data to competitors

## 6. Dispute Resolution
- Disputes should first go through platform mediation (mandatory 30-day period)
- Governing law must be Japanese law for Japan-based transactions

## 7. Price & Fee Changes
- Platform fees cannot increase more than 20% per year without 90 days notice
- Retroactive fee changes are prohibited

## 8. Seller Account & Suspension
- Account suspension requires written notice with specific reasons stated
- Emergency suspension (no notice) only permitted for fraud, safety issues, or illegal activity
- Appeal process must be available within 15 business days
"""

# ── Test contracts ────────────────────────────────────────────────────
# Each contract has planted violations (known_violations) and
# compliant clauses (compliant_clauses) that should NOT be flagged.

TEST_CONTRACTS = [
    {
        "id": "TC01",
        "name": "Severely Non-Compliant Contract",
        "expected_risk_level": "CRITICAL",
        "expected_risk_score_min": 70,
        "contract": """
MARKETPLACE SELLER AGREEMENT

1. PAYMENT TERMS
Platform shall remit payment to Seller within 90 days of order confirmation.
Platform may withhold payments indefinitely if any dispute is pending.
Platform fee is 15% on all transactions with no cap on fee increases.
Fee changes take effect immediately with 1 day notice.
Retroactive fee adjustments may apply to past transactions.

2. INTELLECTUAL PROPERTY
Seller grants Platform an irrevocable, worldwide, royalty-free license to use,
reproduce, distribute, and sublicense all Seller content, trademarks, and product
images for any commercial purpose including advertising and third-party partnerships
without additional consent. Platform shall have joint ownership of seller-generated
reviews and ratings.

3. TERMINATION
Platform may terminate this agreement immediately at any time without notice or reason.
Upon termination, Platform may retain seller funds for up to 180 days.
Seller data will be retained indefinitely by Platform.

4. LIABILITY
Seller assumes full liability for all claims. Platform's liability is limited to $1 USD.
Seller must indemnify Platform against all claims including those resulting from
Platform's own negligence or willful misconduct.

5. DATA
Platform may share Seller's confidential business data with any third parties at its
discretion, including competitors. Data breach notification will occur within 30 days.

6. DISPUTE RESOLUTION
All disputes shall be resolved by binding arbitration in Delaware, USA under US law.

7. ACCOUNT SUSPENSION
Platform may suspend Seller account without notice, reason, or appeal process.
Suspended sellers forfeit all pending payments.
""",
        "known_violations": [
            "payment within 90 days exceeds 60-day rule",
            "payments withheld indefinitely",
            "retroactive fee changes",
            "fee changes with 1 day notice instead of 60 days",
            "sublicensing seller content to third parties",
            "joint ownership of reviews",
            "termination without notice",
            "seller data retained indefinitely instead of deleted within 30 days",
            "platform liability limited to $1",
            "indemnification covers platform's own negligence",
            "business data shared with competitors",
            "data breach notification within 30 days instead of 3-5 days",
            "US law governs instead of Japanese law",
            "suspension without notice or appeal",
        ],
        "compliant_clauses": []  # Nothing compliant here
    },
    {
        "id": "TC02",
        "name": "Mostly Compliant Contract with Subtle Issues",
        "expected_risk_level": "MEDIUM",
        "expected_risk_score_min": 30,
        "expected_risk_score_max": 69,
        "contract": """
MARKETPLACE SELLER AGREEMENT — RAKUTEN ICHIBA

1. PAYMENT TERMS
Platform shall remit payment to Seller within 45 days of confirmed delivery.
Platform fees are 3% for electronics and 5% for apparel. Fee changes require
60 days written notice. No retroactive fee changes permitted.

2. INTELLECTUAL PROPERTY
Seller retains all ownership of their brand, trademarks, and product content.
Platform is granted a non-exclusive license to display Seller content for the
purpose of operating the marketplace only. Sublicensing to third parties requires
Seller's prior written consent.

3. LIABILITY
Each party's liability shall be limited to 12 months of fees paid. Platform
accepts liability for its own gross negligence and willful misconduct.
Consequential damages waiver applies equally to both parties.

4. TERMINATION
Either party may terminate with 30 days written notice. Immediate termination
permitted only for fraud or illegal activity. Seller data deleted within 30 days.
Escrowed funds released within 15 business days.

5. DATA
All data handled per Japan's Act on Protection of Personal Information (APPI).
Data breaches will be reported within 7 days of discovery.
Seller business data will not be shared with competitors.

6. DISPUTE RESOLUTION
Disputes first submitted to platform mediation for 30 days. If unresolved,
disputes governed by Japanese law. Venue: Tokyo District Court.

7. NON-COMPETE
Seller agrees not to operate any competing marketplace business for 2 years
after termination, in Japan or internationally.
""",
        "known_violations": [
            "data breach notification within 7 days instead of 3-5 days",
            "non-compete restricts seller from using other platforms",
        ],
        "compliant_clauses": [
            "payment within 45 days — compliant with 60-day rule",
            "seller retains IP ownership",
            "sublicensing requires written consent",
            "liability cap at 12 months of fees",
            "platform liable for own negligence",
            "30-day termination notice",
            "seller data deleted within 30 days",
            "APPI compliance stated",
            "Japanese law governs",
        ]
    },
    {
        "id": "TC03",
        "name": "Contract with Hidden Fee Trap",
        "expected_risk_level": "HIGH",
        "expected_risk_score_min": 50,
        "contract": """
SELLER AGREEMENT — MERCARI SHOPS

1. PAYMENT
Mercari shall pay seller within 30 days of transaction confirmation. Standard
platform fee is 10%. Platform reserves the right to introduce new service fees
and surcharges at any time with 14 days notice. Fee increases of any magnitude
are permitted with 30 days notice.

2. INTELLECTUAL PROPERTY
Seller retains ownership of all original content. Platform may use seller content
for marketplace display and internal analytics. No sublicensing without consent.

3. TERMINATION
Seller may terminate with 30 days notice. Platform may terminate with 30 days
notice for any reason, or immediately for fraud or safety violations.
Seller funds released within 15 business days of termination.
Seller data deleted within 30 days.

4. LIABILITY
Platform liability capped at 6 months of fees paid. Seller liability capped at
12 months. Platform disclaims liability for indirect damages but accepts liability
for direct damages from its own misconduct.

5. DATA
Data processed per APPI. Security breach notification within 3 days.
Seller data not sold to third parties.

6. DISPUTES
Mediation required for 30 days before escalation. Japanese law applies.

7. SUSPENSION
Account suspension requires written notice with reasons. Emergency suspension
for fraud only. Appeal available within 15 business days.
""",
        "known_violations": [
            "fee increases of any magnitude with only 30 days notice violates 20% cap rule",
            "new service fees and surcharges with only 14 days notice",
            "platform liability capped at 6 months instead of allowed 12 months for seller",
        ],
        "compliant_clauses": [
            "payment within 30 days — compliant",
            "seller retains IP",
            "no sublicensing without consent",
            "seller funds released within 15 business days",
            "seller data deleted within 30 days",
            "data breach notification within 3 days",
            "30-day mediation required",
            "Japanese law applies",
            "suspension requires written notice",
            "appeal within 15 business days",
        ]
    }
]


# ── Minimal compliance analyzer (no LangChain dependency) ────────────
def analyze_contract(contract_text: str, client: Anthropic) -> dict:
    prompt = f"""You are an expert legal AI specializing in Japanese marketplace compliance.

Analyze the following seller contract against Japanese marketplace rules and return ONLY valid JSON.

MARKETPLACE RULES:
{MARKETPLACE_RULES}

CONTRACT:
{contract_text}

Return this exact JSON structure:
{{
  "risk_score": <0-100, 0=fully compliant, 100=severely non-compliant>,
  "risk_level": "<CRITICAL|HIGH|MEDIUM|LOW>",
  "summary": "<2-3 sentence summary>",
  "critical_issues": [
    {{"clause": "...", "issue": "...", "rule_violated": "..."}}
  ],
  "high_issues": [
    {{"clause": "...", "issue": "...", "rule_violated": "..."}}
  ],
  "medium_issues": [
    {{"clause": "...", "issue": "...", "rule_violated": "..."}}
  ],
  "compliant_items": ["..."]
}}

Return ONLY valid JSON, no other text."""

    r = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}]
    )
    raw = r.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())


# ── Scoring ───────────────────────────────────────────────────────────

def fuzzy_match(text: str, keyword: str, threshold: float = 0.35) -> bool:
    text_lower = text.lower()
    words = [w for w in keyword.lower().split() if len(w) > 3]
    if not words:
        return keyword.lower() in text_lower
    matches = sum(1 for w in words if w in text_lower)
    return matches / len(words) >= threshold


def score_violation_detection(result: dict, known_violations: list) -> dict:
    """What % of known violations did the system catch?"""
    if not known_violations:
        return {"detection_rate": 1.0, "caught": 0, "total": 0}

    all_issues = []
    for issue in result.get("critical_issues", []):
        all_issues.append(f"{issue.get('clause','')} {issue.get('issue','')}")
    for issue in result.get("high_issues", []):
        all_issues.append(f"{issue.get('clause','')} {issue.get('issue','')}")
    for issue in result.get("medium_issues", []):
        all_issues.append(f"{issue.get('clause','')} {issue.get('issue','')}")

    issues_text = " ".join(all_issues)
    caught = sum(1 for v in known_violations if fuzzy_match(issues_text, v))

    return {
        "detection_rate": round(caught / len(known_violations), 3),
        "caught": caught,
        "total": len(known_violations)
    }


def score_false_positives(result: dict, compliant_clauses: list) -> dict:
    """What % of compliant clauses were wrongly flagged as issues?"""
    if not compliant_clauses:
        return {"false_positive_rate": 0.0, "wrongly_flagged": 0, "total": 0}

    all_issues = []
    for issue in result.get("critical_issues", []):
        all_issues.append(f"{issue.get('clause','')} {issue.get('issue','')}")
    for issue in result.get("high_issues", []):
        all_issues.append(f"{issue.get('clause','')} {issue.get('issue','')}")
    issues_text = " ".join(all_issues)

    wrongly_flagged = sum(
        1 for c in compliant_clauses
        if fuzzy_match(issues_text, c.split("—")[0].strip())
    )

    return {
        "false_positive_rate": round(wrongly_flagged / len(compliant_clauses), 3),
        "wrongly_flagged": wrongly_flagged,
        "total": len(compliant_clauses)
    }


def score_risk_level(result: dict, expected_level: str,
                     expected_min: int, expected_max: int = 100) -> dict:
    actual_level = result.get("risk_level", "")
    actual_score = result.get("risk_score", 0)

    level_correct = actual_level == expected_level
    score_in_range = expected_min <= actual_score <= expected_max

    # Partial credit: adjacent levels
    level_order = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    try:
        exp_idx = level_order.index(expected_level)
        act_idx = level_order.index(actual_level)
        level_distance = abs(exp_idx - act_idx)
    except ValueError:
        level_distance = 2

    return {
        "level_correct": level_correct,
        "score_in_range": score_in_range,
        "expected_level": expected_level,
        "actual_level": actual_level,
        "expected_score_min": expected_min,
        "actual_score": actual_score,
        "level_distance": level_distance,
    }


# ── Main ──────────────────────────────────────────────────────────────

def print_banner():
    print(f"""
{BLUE}{BOLD}╔══════════════════════════════════════════════════════╗
║   Marketplace Compliance Checker — Evaluation v1.0   ║
║   3 contracts · Known violation ground truth         ║
╚══════════════════════════════════════════════════════╝{RESET}
""")


def run_evaluation():
    print_banner()

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print(f"{RED}ANTHROPIC_API_KEY not set.{RESET}")
        return

    client = Anthropic(api_key=api_key)
    all_results = []
    aggregate = {
        "detection_rate": [],
        "false_positive_rate": [],
        "risk_level_correct": [],
    }

    for i, tc in enumerate(TEST_CONTRACTS, 1):
        print(f"{BLUE}[{i}/{len(TEST_CONTRACTS)}] {tc['name']}{RESET}")
        print(f"  Expected risk: {tc['expected_risk_level']} | Violations planted: {len(tc['known_violations'])}")

        start = time.time()
        try:
            result = analyze_contract(tc["contract"], client)
        except Exception as e:
            print(f"  {RED}Error: {e}{RESET}\n")
            continue
        elapsed = round(time.time() - start, 1)

        detection = score_violation_detection(result, tc["known_violations"])
        fp = score_false_positives(result, tc["compliant_clauses"])
        risk = score_risk_level(
            result,
            tc["expected_risk_level"],
            tc["expected_risk_score_min"],
            tc.get("expected_risk_score_max", 100)
        )

        # False positive score (inverted — lower FP rate = better)
        fp_score = 1.0 - fp["false_positive_rate"]

        def bar(score):
            filled = int(score * 15)
            color = GREEN if score >= 0.8 else YELLOW if score >= 0.6 else RED
            return f"{color}{'█' * filled}{'░' * (15 - filled)} {score:.2f}{RESET}"

        risk_color = GREEN if risk["level_correct"] else (YELLOW if risk["level_distance"] <= 1 else RED)
        print(f"  Violation detection  {bar(detection['detection_rate'])}  ({detection['caught']}/{detection['total']} caught)")
        print(f"  False positive rate  {bar(fp_score)}  ({fp['wrongly_flagged']} wrongly flagged)")
        print(f"  Risk level           {risk_color}{risk['actual_level']}{RESET} (expected {risk['expected_level']}) — {'✓' if risk['level_correct'] else '~' if risk['level_distance'] <= 1 else '✗'}")
        print(f"  Risk score           {result.get('risk_score', '?')}/100 (expected ≥{tc['expected_risk_score_min']})")
        print(f"  ⏱  {elapsed}s\n")

        aggregate["detection_rate"].append(detection["detection_rate"])
        aggregate["false_positive_rate"].append(fp["false_positive_rate"])
        aggregate["risk_level_correct"].append(1.0 if risk["level_correct"] else 0.5 if risk["level_distance"] <= 1 else 0.0)

        all_results.append({
            "test_case": tc["id"],
            "name": tc["name"],
            "elapsed_seconds": elapsed,
            "scores": {
                "violation_detection_rate": detection["detection_rate"],
                "false_positive_rate": fp["false_positive_rate"],
                "risk_level_correct": risk["level_correct"],
                "risk_level_distance": risk["level_distance"],
                "risk_score_in_range": risk["score_in_range"],
            },
            "details": {
                "detection": detection,
                "false_positives": fp,
                "risk_assessment": risk,
            },
            "model_output": {
                "risk_level": result.get("risk_level"),
                "risk_score": result.get("risk_score"),
                "critical_count": len(result.get("critical_issues", [])),
                "high_count": len(result.get("high_issues", [])),
                "medium_count": len(result.get("medium_issues", [])),
            }
        })

        time.sleep(2)

    # ── Summary ───────────────────────────────────────────────────────
    def avg(lst): return round(sum(lst) / len(lst), 3) if lst else 0

    det_avg = avg(aggregate["detection_rate"])
    fp_avg = avg(aggregate["false_positive_rate"])
    level_avg = avg(aggregate["risk_level_correct"])
    fp_score_avg = 1.0 - fp_avg
    overall = avg([det_avg, fp_score_avg, level_avg])

    print(f"{BLUE}{BOLD}{'═' * 56}{RESET}")
    print(f"{BOLD}  AGGREGATE RESULTS — {len(TEST_CONTRACTS)} contracts{RESET}")
    print(f"{BLUE}{BOLD}{'═' * 56}{RESET}\n")

    def summary_bar(score):
        filled = int(score * 20)
        color = GREEN if score >= 0.8 else YELLOW if score >= 0.6 else RED
        status = "✓ GOOD" if score >= 0.8 else "~ OK" if score >= 0.6 else "✗ NEEDS WORK"
        return f"{color}{score:.3f}  [{'█'*filled}{'░'*(20-filled)}]  {status}{RESET}"

    print(f"  {BOLD}{'Violation Detection Rate':<28}{RESET} {summary_bar(det_avg)}")
    print(f"  {BOLD}{'False Positive Score (inv.)':<28}{RESET} {summary_bar(fp_score_avg)}")
    print(f"  {BOLD}{'Risk Level Accuracy':<28}{RESET} {summary_bar(level_avg)}")
    print(f"\n  {BOLD}{'Overall Score':<28}{RESET} {GREEN if overall >= 0.8 else YELLOW if overall >= 0.6 else RED}{overall:.3f}{RESET}")
    print(f"  {BOLD}{'Contracts Tested':<28}{RESET} {len(TEST_CONTRACTS)} (human-labeled violations)")
    print(f"  {BOLD}{'Model':<28}{RESET} claude-haiku-4-5-20251001")
    print(f"  {BOLD}{'Evaluated':<28}{RESET} {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")

    output = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "version": "1.0",
        "model": "claude-haiku-4-5-20251001",
        "contracts_tested": len(TEST_CONTRACTS),
        "ground_truth_type": "human-labeled violations",
        "aggregate_scores": {
            "violation_detection_rate": det_avg,
            "false_positive_rate": fp_avg,
            "false_positive_score": fp_score_avg,
            "risk_level_accuracy": level_avg,
        },
        "overall_score": overall,
        "per_test": all_results
    }

    with open("eval_results.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"  {GREEN}✓ Results saved to eval_results.json{RESET}")
    print(f"\n  {YELLOW}Add to README:{RESET}")
    print(f"  Violation Detection: {det_avg:.1%} | False Positive Score: {fp_score_avg:.1%} | Risk Level Accuracy: {level_avg:.1%}\n")


if __name__ == "__main__":
    run_evaluation()
