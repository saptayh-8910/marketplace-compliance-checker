
---

```markdown
# Marketplace Compliance Checker 🛡️🤖

This document analyzer is an autonomous, multi-agent compliance and security auditing system engineered to analyze marketplace product listings (Shopify, Amazon Japan, Mercari) before publication. 

By leveraging agentic orchestration, the system evaluates product images, promotional text overlays, and listing metadata against e-commerce platform guidelines, legal advertising frameworks, and consumer safety rules—providing automated scoring, risk maps, and remediation feedback.

---

## 🏗️ System Architecture & Data Flow

This platform utilizes a specialized multi-agent hierarchy to ensure separate areas of concern, high-precision retrieval, and deterministic evaluation loops.

```text
  [ User Listing Input ] (Images, Metadata, System Settings)
            │
            ▼
┌────────────────────────────────────────────────────────┐
│            ORCHESTRATOR / SUPERVISOR AGENT             │
└───────────┬────────────────────────┬───────────────────┘
            │                        │
            ▼                        ▼
┌───────────────────────┐┌───────────────────────┐
│     VISION AUDIT      ││    SEMANTIC AUDIT     │
│         AGENT         ││         AGENT         │
├───────────────────────┤├───────────────────────┤
│ • OCR Price-Tags      ││ • Risk Scanning       │
│ • Aspect-Ratio / Crop ││ • Guideline Retrieval │
│ • Platform Violations ││ • Keyword Verification │
└───────────┬───────────┘└───────────┬───────────┘
            │                        │
            └───────────┬────────────┘
                        │ (Raw Audits)
                        ▼
┌────────────────────────────────────────────────────────┐
│             COMPLIANCE SYNTHESIS ENGINE                │
├────────────────────────────────────────────────────────┤
│  • Computes Structural Scores & Platform Fit          │
│  • Flags Regulatory Violations (e.g., False Claims)   │
│  • Generates Remediation Copy & A/B Action Insights    │
└───────────────────────┬────────────────────────────────┘
                        │
                        ▼
             [ JSON Audit Report / UI ]

```

---

## ✨ Core Features

* **Multi-Agent Auditing Core:** Distributed execution split between visual asset inspection (Vision LLM) and listing copy verification (Semantic LLM).
* **Platform-Specific Rule Engines:** Tailored validation layers matching guidelines for **Amazon JP**, **Mercari**, and international storefronts (Shopify, Etsy).
* **Visual Overlay Processing (OCR):** Detects promotional text overlays, text-to-canvas hierarchies, price tags, and call-to-action (CTA) clarity rules.
* **Structured Risk Matrix:** Outputs deterministic evaluation logs including specific line-by-line violation triggers and actionable fix suggestions.

---

## 🛠️ Technical Stack

* **Orchestration Framework:** Python-native agentic framework (Google Antigravity / Google Stitch / LangChain / n8n)
* **Inference Engines:** Anthropic Claude (via `ChatAnthropic`) for structural semantic synthesis and high-token reasoning.
* **Vector Database:** ChromaDB (Dense Retrieval via `intfloat/multilingual-e5-large`) for loading local platform legal frameworks.
* **UI Interface:** Streamlit (Cached data pipelines, stateful multithreaded chat interface).

---

## 🚀 Getting Started

### 1. Prerequisites & Environment Setup

Clone the repository and ensure you have Python 3.11+ installed:

```bash
git clone [https://github.com/saptayh-8910/marketplace-compliance-checker.git](https://github.com/saptayh-8910/marketplace-compliance-checker.git)
cd marketplace-compliance-checker
pip install -r requirements.txt

```

Create a `.env` file in the root directory and append your secure provider credentials:

```env
ANTHROPIC_API_KEY=your_claude_api_key_here
EMBEDDING_MODEL_NAME=intfloat/multilingual-e5-large
ENV_STATE=production

```

### 2. Running the Data Pipeline & Local Indexing

Before initiating the compliance checker, load the relevant regulatory guideline documentation into the vector system storage:

```bash
python ingest.py --docs_dir ./docs

```

### 3. Launching the Auditing UI Dashboard

Start the local Streamlit application to upload your product imagery, price variations, and catalog meta strings:

```bash
streamlit run app.py

```

---

## 📊 Evaluation & Verification (LLMOps Baseline)

AgenticSentry measures accuracy using rigorous multi-turn evaluation frameworks. The underlying scoring model measures **Faithfulness**, **Answer Relevancy**, and **Context Precision**.

The execution history and benchmark baseline report can be generated using our test harness:

```bash
python evaluate.py

```

Current System Benchmarks (`ragas_results.json`):

* **Faithfulness (Groundedness):** `0.890` — Low risk of hallucinating non-existent compliance violations.
* **Context Precision:** Testing implementation under `v2.0` architecture optimizations.
---

## 📝 License

Distributed under the MIT License. See `LICENSE` for more information.
