# Marketplace Compliance Checker

> Upload a seller contract PDF → AI checks it against Rakuten, Mercari, and Amazon Japan rules → get a risk score, flagged clauses, and fix recommendations.

**Built as Level 2B of an AI Engineer → AI Product Manager roadmap targeting the Japanese tech market.**

---

## Live Demo

⚖️ **[Try the Compliance Checker →](https://marketplace-compliance-checker.streamlit.app)**

---

## Evaluation

Tested on 3 contracts with planted violations ranging from subtle to severe.

| Metric | Score |
|---|---|
| Violation Detection Rate | **100%** |
| False Positive Score | **93.3%** |
| Risk Level Accuracy | **83.3%** |

Evaluation script: `evaluate_compliance_checker.py` · Results: `eval_results.json`

---

## What It Does

Upload a seller agreement PDF. The app extracts the text, retrieves the most relevant compliance rules using semantic search, and asks Claude to flag violations, score overall risk, and recommend specific fixes.

Output for each contract:
- Risk score (0–100) and risk level (LOW / MEDIUM / HIGH / CRITICAL)
- Critical, high, and medium issues — each with the specific clause, the rule it violates, and what needs to change
- List of compliant items
- Downloadable plain-text report

---

## How It Works

```
Upload contract PDF
        ↓
Extract text (PyPDFLoader)
        ↓
Semantic search over JP marketplace rules (ChromaDB + multilingual-e5-large)
        ↓
Claude Haiku analyzes contract against retrieved rules
        ↓
Structured JSON output: risk score + flagged clauses + recommendations
        ↓
Streamlit UI renders findings by severity tab
```

---

## Rules Knowledge Base

The app checks contracts against 12 categories of Japanese marketplace rules covering:

- Payment terms (60-day rule, fee change notice periods)
- Liability & indemnification caps
- IP ownership and sublicensing restrictions
- Contract termination and data deletion requirements
- APPI data privacy compliance
- Dispute resolution (Japanese law, Tokyo District Court)
- Account suspension and appeal rights
- Platform-specific rules for Rakuten, Mercari, and Amazon Japan

Legal framework references: Japan Commercial Code, Product Liability Act, APPI, Payment Services Act, Provider Liability Limitation Act.

---

## Japan Market Relevance

Seller agreement disputes are common in Japan's marketplace ecosystem. This tool targets:
- **Mercari Shops** sellers reviewing platform agreements
- **Rakuten Ichiba** merchants checking third-party vendor contracts
- **Amazon Japan** sellers verifying fulfillment agreements
- Legal and compliance teams at JP fintech companies (MoneyForward, PayPay)

---

## Setup

```bash
git clone https://github.com/saptayh-8910/marketplace-compliance-checker.git
cd marketplace-compliance-checker

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

Create `.env`:
```
ANTHROPIC_API_KEY=your_key_here
```

Run:
```bash
streamlit run app.py
```

---

## Stack

| Layer | Component |
|---|---|
| LLM | Claude Haiku (`claude-haiku-4-5-20251001`) |
| Embeddings | `intfloat/multilingual-e5-large` |
| Vector store | ChromaDB (local, persistent) |
| PDF extraction | LangChain PyPDFLoader |
| Framework | LangChain + Streamlit |
| Rules base | Hardcoded JP marketplace rules (12 categories) |

---

## Design Decisions

**Why RAG over the rules instead of putting them all in the prompt?**
The full rules knowledge base exceeds what fits cleanly in a single prompt context for structured output. RAG retrieves the 8 most relevant rule chunks for each contract, keeping the prompt focused and reducing noise in the output.

**Why Claude Haiku?**
Speed and cost — compliance checks need to feel instant. Haiku handles structured JSON extraction reliably at roughly 1/15th the cost of Sonnet. The rules retrieval does the heavy lifting; the LLM just needs to match text against retrieved context.

**Why not use an external legal API?**
The rules are hardcoded intentionally. Connecting to live legal databases adds dependency and cost. For a portfolio demo and MVP, a well-curated static knowledge base is sufficient and fully auditable.

---

## Limitations

- Rules knowledge base is static (last updated June 2026) — real marketplace terms change
- PDF extraction quality depends on the PDF being text-based, not scanned
- Not legal advice — output is for reference only
- English contracts only (Japanese contract support planned)

---

## Project Structure

```
marketplace-compliance-checker/
├── app.py                        # Main Streamlit application
├── evaluate_compliance_checker.py # Evaluation script
├── eval_results.json             # Latest evaluation scores
├── requirements.txt
└── README.md
```

---

*Part of an AI Engineer → Senior AI Product Manager portfolio · [View full roadmap](https://github.com/saptayh-8910/rag-assistant)*
