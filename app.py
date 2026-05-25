import os
import json
import tempfile
import streamlit as st
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import shutil

load_dotenv()

# ── Page config ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Marketplace Compliance Checker",
    page_icon="⚖️",
    layout="wide"
)

st.markdown("""
<style>
    .main { padding-top: 0.5rem; }
    .header-box {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        color: white;
    }
    .header-box h1 { color: white; margin: 0; font-size: 1.8rem; }
    .header-box p { color: rgba(255,255,255,0.75); margin: 4px 0 0; font-size: 0.95rem; }
    .risk-critical { background: #FFEBEE; border-left: 5px solid #D32F2F; padding: 12px 16px; border-radius: 0 8px 8px 0; margin: 8px 0; }
    .risk-high { background: #FFF3E0; border-left: 5px solid #F57C00; padding: 12px 16px; border-radius: 0 8px 8px 0; margin: 8px 0; }
    .risk-medium { background: #FFFDE7; border-left: 5px solid #F9A825; padding: 12px 16px; border-radius: 0 8px 8px 0; margin: 8px 0; }
    .risk-low { background: #E8F5E9; border-left: 5px solid #388E3C; padding: 12px 16px; border-radius: 0 8px 8px 0; margin: 8px 0; }
    .score-box { text-align: center; padding: 20px; border-radius: 12px; margin: 10px 0; }
    .compliant-item { background: #E8F5E9; border-radius: 6px; padding: 8px 12px; margin: 4px 0; font-size: 14px; color: #2E7D32; }
    .section-header { font-size: 16px; font-weight: 600; color: #1a1a2e; margin: 16px 0 8px; border-bottom: 2px solid #0f3460; padding-bottom: 4px; }
</style>
""", unsafe_allow_html=True)

# ── Marketplace rules knowledge base ─────────────────────────────────
MARKETPLACE_RULES = """
# Rakuten & Mercari Marketplace Seller Compliance Rules

## 1. Payment Terms
- Payment to sellers must be processed within 60 days of confirmed delivery
- Platform fees must be clearly stated (Rakuten: 2-7% depending on category, Mercari: 10%)
- Late payment penalties must not exceed 14.6% annual interest per Japanese Commercial Code
- Currency must be specified as Japanese Yen (JPY) for domestic transactions
- Escrow arrangements must comply with Japan Payment Services Act

## 2. Liability & Indemnification
- Seller liability cap must not exceed 12 months of total fees paid
- Platform cannot disclaim liability for gross negligence or willful misconduct under Japanese law
- Product liability must comply with Japan Product Liability Act (PL Act)
- Consequential damages waiver must be mutual (cannot apply only to platform)
- Force majeure clauses must list specific events and cannot be overly broad

## 3. Intellectual Property (IP)
- Seller must retain ownership of their brand and product IP
- Platform license to use seller content must be limited to marketplace operations only
- Sublicensing of seller content to third parties requires explicit written consent
- Platform cannot claim ownership of seller-generated reviews or ratings
- Takedown procedures must comply with Japan's Provider Liability Limitation Act

## 4. Prohibited Items & Content
- All listings must comply with Japan's Act against Unjustifiable Premiums and Misleading Representations
- Medical devices require approval under Japan's Pharmaceuticals and Medical Devices Act (PMDA)
- Food items must comply with Japan's Food Sanitation Act
- Alcohol sales require proper licensing under Japan's Liquor Tax Act
- Counterfeit goods are strictly prohibited and grounds for immediate termination

## 5. Contract Term & Termination
- Minimum notice period for contract termination must be 30 days (Rakuten standard: 60 days)
- Immediate termination only permitted for material breach, fraud, or illegal activity
- Seller data must be returned or deleted within 30 days of contract termination
- Funds held in escrow must be released within 15 business days of termination
- Non-compete clauses cannot restrict seller from using other Japanese platforms

## 6. Data Privacy & Security
- All seller and buyer data must be handled per Japan's Act on Protection of Personal Information (APPI)
- Cross-border data transfers require adequate protection measures
- Data breach notification must occur within 3-5 days per Japan APPI 2022 amendment
- Platform cannot sell seller business data to competitors
- Security requirements must meet PCI DSS standards for payment data

## 7. Dispute Resolution
- Disputes should first go through platform mediation (mandatory 30-day period)
- Arbitration must be available as alternative to litigation
- Governing law must be Japanese law for Japan-based transactions
- Venue for litigation should be Tokyo District Court (standard for major platforms)
- Class action waivers are generally enforceable in Japan

## 8. Price & Fee Changes
- Platform fees cannot increase more than 20% per year without 90 days notice
- Any fee changes require written notice at least 60 days in advance
- Promotional fee reductions can be withdrawn with 14 days notice
- Retroactive fee changes are prohibited

## 9. Seller Account & Suspension
- Account suspension requires written notice with specific reasons stated
- Emergency suspension (no notice) only permitted for fraud, safety issues, or illegal activity
- Appeal process must be available within 15 business days
- Wrongful suspension entitles seller to compensation for lost sales

## 10. Rakuten-Specific Rules (2024-2026)
- Rakuten Ichiba sellers must maintain minimum 4.0 store rating
- R-Pay integration is mandatory for Rakuten mall sellers
- Rakuten Super Points allocation: minimum 1% on all transactions
- Super Sale participation is optional but fee structure must be pre-agreed

## 11. Mercari-Specific Rules (2024-2026)
- Mercari takes 10% platform fee on all sales
- Sellers must ship within 3 days of purchase confirmation
- Mercari EAZY logistics integration is optional
- Price changes after purchase confirmation are prohibited
- Mercari Shops sellers have separate terms from C2C marketplace

## 12. Amazon Japan-Specific Rules (2024-2026)
- Amazon takes 8-15% referral fee depending on category
- FBA (Fulfillment by Amazon) terms are separate from seller agreements
- Buy Box eligibility requires competitive pricing policy compliance
- Amazon can remove listings without prior notice for policy violations
- Selling Partner API terms apply for programmatic access
"""

# ── Load rules into vector store ──────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_rules_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")
    persist_dir = "./chroma_rules"

    if os.path.exists(persist_dir) and os.listdir(persist_dir):
        return Chroma(persist_directory=persist_dir, embedding_function=embeddings)

    # Write rules to temp file and load
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
        f.write(MARKETPLACE_RULES)
        temp_path = f.name

    loader = TextLoader(temp_path, encoding='utf-8')
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=80)
    chunks = splitter.split_documents(docs)

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir
    )
    os.unlink(temp_path)
    return vectordb

@st.cache_resource
def get_llm():
    return ChatAnthropic(
        model="claude-haiku-4-5-20251001",
        temperature=0,
        max_tokens=4096
    )

# ── Extract text from uploaded PDF ───────────────────────────────────
def extract_contract_text(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name
    try:
        loader = PyPDFLoader(tmp_path)
        pages = loader.load()
        text = "\n\n".join(p.page_content for p in pages)
        return text, len(pages)
    finally:
        os.unlink(tmp_path)

# ── Core compliance analysis ──────────────────────────────────────────
def analyze_contract(contract_text, platform, llm, vectordb):
    # Retrieve relevant rules
    retriever = vectordb.as_retriever(search_kwargs={"k": 8})
    relevant_rules = retriever.invoke(f"{platform} marketplace seller compliance rules")
    rules_context = "\n\n".join(d.page_content for d in relevant_rules)

    prompt = f"""You are an expert legal AI specializing in Japanese marketplace compliance for {platform}.

Analyze the following vendor/seller contract against {platform}'s marketplace rules and Japanese law requirements.

MARKETPLACE RULES & REQUIREMENTS:
{rules_context}

CONTRACT TO ANALYZE:
{contract_text[:6000]}

Provide a detailed compliance analysis in the following JSON format:
{{
  "risk_score": <integer 0-100, where 0=fully compliant, 100=severely non-compliant>,
  "risk_level": "<CRITICAL|HIGH|MEDIUM|LOW>",
  "platform": "{platform}",
  "summary": "<2-3 sentence executive summary of the contract's compliance status>",
  "critical_issues": [
    {{
      "clause": "<specific clause or section name>",
      "issue": "<what the problem is>",
      "rule_violated": "<which rule it violates>",
      "recommendation": "<specific fix required>"
    }}
  ],
  "high_issues": [
    {{
      "clause": "<clause name>",
      "issue": "<problem>",
      "rule_violated": "<rule>",
      "recommendation": "<fix>"
    }}
  ],
  "medium_issues": [
    {{
      "clause": "<clause name>",
      "issue": "<problem>",
      "rule_violated": "<rule>",
      "recommendation": "<fix>"
    }}
  ],
  "compliant_items": [
    "<item that IS compliant with marketplace rules>"
  ],
  "overall_recommendation": "<1-2 sentences on whether to sign, negotiate, or reject this contract>"
}}

Return ONLY valid JSON, no other text.
"""

    response = llm.invoke(prompt)
    raw = response.content.strip()

    # Clean JSON
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    return json.loads(raw)

# ── Sample contract for demo ──────────────────────────────────────────
SAMPLE_CONTRACT = """
MARKETPLACE SELLER AGREEMENT

This Seller Agreement ("Agreement") is entered into between TechMart Platform Inc. ("Platform") 
and the undersigned seller ("Seller").

1. PAYMENT TERMS
Platform shall remit payment to Seller within 90 days of order confirmation. Platform reserves 
the right to withhold payments indefinitely if any dispute is pending. Platform fee is 15% on 
all transactions with no cap on fee increases.

2. INTELLECTUAL PROPERTY
Seller grants Platform an irrevocable, worldwide, royalty-free license to use, reproduce, 
distribute, and sublicense all Seller content, trademarks, and product images for any 
commercial purpose including advertising and third-party partnerships without additional consent.
Platform shall have joint ownership of any seller-generated reviews and ratings.

3. TERMINATION
Platform may terminate this agreement immediately at any time without notice or reason.
Upon termination, Platform may retain seller funds for up to 180 days for "security purposes."
Seller data will be retained indefinitely by Platform.

4. LIABILITY
Seller assumes full liability for all claims. Platform's liability is limited to $1 USD 
regardless of the nature or extent of damages. Seller must indemnify Platform against all 
claims including those resulting from Platform's own negligence.

5. DISPUTE RESOLUTION
All disputes shall be resolved by binding arbitration in Delaware, USA under US law.
Class action rights are waived. Seller waives right to jury trial and any appeal.

6. DATA
Platform may share Seller's confidential business data with any third parties at its discretion.
Seller consents to Platform using sales data for competitive intelligence purposes.

7. FEES
Platform may change fees at any time with 1 day notice. Retroactive fee changes may apply
to past transactions at Platform's sole discretion.

8. ACCOUNT SUSPENSION  
Platform may suspend Seller account without notice, reason, or appeal process.
Suspended sellers forfeit all pending payments.
"""

# ══ MAIN UI ═══════════════════════════════════════════════════════════

st.markdown("""
<div class="header-box">
    <h1>⚖️ Marketplace Seller Compliance Checker</h1>
    <p>Check vendor contracts against Rakuten, Mercari & Amazon Japan rules · Powered by Claude AI · 日本語対応</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    platform = st.selectbox(
        "Target Marketplace",
        ["Rakuten Ichiba", "Mercari", "Amazon Japan", "General JP Marketplace"],
        help="Select the marketplace whose rules to check against"
    )

    st.divider()
    st.markdown("### 📋 What We Check")
    checks = [
        "💳 Payment terms & timing",
        "⚖️ Liability & indemnification",
        "🏷️ IP rights & content license",
        "🚫 Prohibited items",
        "📅 Termination conditions",
        "🔒 Data privacy (APPI)",
        "💰 Fee structure & changes",
        "🏛️ Dispute resolution",
        "📊 Account suspension rules",
    ]
    for c in checks:
        st.markdown(f"<div style='font-size:13px; padding: 3px 0;'>{c}</div>", unsafe_allow_html=True)

    st.divider()
    st.markdown("### 🏛️ Legal Framework")
    st.markdown("""
    <div style='font-size:12px; color: #666;'>
    • Japan Commercial Code<br>
    • Product Liability Act (PL Act)<br>
    • APPI (Personal Information Protection)<br>
    • Payment Services Act<br>
    • Provider Liability Limitation Act<br>
    • Act on Unjustifiable Premiums
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.caption("⚖️ For reference only. Not legal advice.")
    st.caption("Consult a qualified Japanese attorney.")

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 📄 Upload Contract")
    uploaded_file = st.file_uploader(
        "Upload seller contract PDF",
        type=["pdf"],
        help="Upload a PDF vendor or seller agreement to analyze"
    )

    st.markdown("**Or use our sample risky contract:**")
    use_sample = st.button("⚡ Analyze Sample Contract", use_container_width=True,
                           help="Uses a sample contract with intentionally problematic clauses")

    if uploaded_file or use_sample:
        if st.button("🔍 Run Compliance Check", type="primary", use_container_width=True) or use_sample:
            with st.spinner("Loading compliance rules..."):
                vectordb = get_rules_vectorstore()
                llm = get_llm()

            if use_sample and not uploaded_file:
                contract_text = SAMPLE_CONTRACT
                num_pages = 1
                st.info("Using sample contract with intentionally problematic clauses")
            else:
                with st.spinner("Extracting contract text..."):
                    contract_text, num_pages = extract_contract_text(uploaded_file)
                st.success(f"✓ Extracted {num_pages} pages")

            with st.spinner(f"Analyzing against {platform} compliance rules..."):
                try:
                    result = analyze_contract(contract_text, platform, llm, vectordb)
                    st.session_state.result = result
                    st.session_state.contract_text = contract_text
                except json.JSONDecodeError as e:
                    st.error(f"Could not parse analysis result. Try again.")
                except Exception as e:
                    st.error(f"Analysis error: {e}")

with col2:
    st.markdown("### 📊 Analysis Results")

    if "result" not in st.session_state:
        st.info("Upload a contract and click 'Run Compliance Check' to see results here.")
        st.markdown("""
        **How it works:**
        1. Upload your seller agreement PDF
        2. Select the target marketplace
        3. AI analyzes against JP marketplace rules
        4. Get risk score + flagged clauses + recommendations
        """)
    else:
        result = st.session_state.result
        score = result.get("risk_score", 50)
        risk_level = result.get("risk_level", "MEDIUM")

        # Risk score display
        score_color = "#D32F2F" if score >= 70 else "#F57C00" if score >= 40 else "#388E3C"
        score_bg = "#FFEBEE" if score >= 70 else "#FFF3E0" if score >= 40 else "#E8F5E9"
        score_emoji = "🔴" if score >= 70 else "🟡" if score >= 40 else "🟢"

        st.markdown(f"""
        <div class="score-box" style="background:{score_bg}; border: 2px solid {score_color}">
            <div style="font-size:48px; font-weight:800; color:{score_color}">{score}</div>
            <div style="font-size:14px; color:{score_color}; font-weight:600">Risk Score / 100</div>
            <div style="font-size:20px; margin-top:8px">{score_emoji} {risk_level} RISK</div>
        </div>
        """, unsafe_allow_html=True)

        # Progress bar
        st.progress(score / 100)
        st.markdown(f"**{result.get('platform', platform)} Compliance Analysis**")
        st.markdown(f"_{result.get('summary', '')}_")
        st.markdown(f"**Recommendation:** {result.get('overall_recommendation', '')}")

# ── Detailed findings ──────────────────────────────────────────────────
if "result" in st.session_state:
    result = st.session_state.result
    st.divider()
    st.markdown("## 📋 Detailed Findings")

    tab1, tab2, tab3, tab4 = st.tabs([
        f"🔴 Critical ({len(result.get('critical_issues', []))})",
        f"🟠 High ({len(result.get('high_issues', []))})",
        f"🟡 Medium ({len(result.get('medium_issues', []))})",
        f"✅ Compliant ({len(result.get('compliant_items', []))})"
    ])

    with tab1:
        issues = result.get("critical_issues", [])
        if issues:
            for i, issue in enumerate(issues, 1):
                st.markdown(f"""<div class="risk-critical">
                    <strong>#{i} {issue.get('clause', 'Unknown clause')}</strong><br>
                    🚨 <strong>Issue:</strong> {issue.get('issue', '')}<br>
                    📋 <strong>Rule violated:</strong> {issue.get('rule_violated', '')}<br>
                    ✏️ <strong>Fix required:</strong> {issue.get('recommendation', '')}
                </div>""", unsafe_allow_html=True)
        else:
            st.success("No critical issues found!")

    with tab2:
        issues = result.get("high_issues", [])
        if issues:
            for i, issue in enumerate(issues, 1):
                st.markdown(f"""<div class="risk-high">
                    <strong>#{i} {issue.get('clause', 'Unknown clause')}</strong><br>
                    ⚠️ <strong>Issue:</strong> {issue.get('issue', '')}<br>
                    📋 <strong>Rule violated:</strong> {issue.get('rule_violated', '')}<br>
                    ✏️ <strong>Recommendation:</strong> {issue.get('recommendation', '')}
                </div>""", unsafe_allow_html=True)
        else:
            st.success("No high-severity issues found!")

    with tab3:
        issues = result.get("medium_issues", [])
        if issues:
            for i, issue in enumerate(issues, 1):
                st.markdown(f"""<div class="risk-medium">
                    <strong>#{i} {issue.get('clause', 'Unknown clause')}</strong><br>
                    ⚡ <strong>Issue:</strong> {issue.get('issue', '')}<br>
                    📋 <strong>Rule violated:</strong> {issue.get('rule_violated', '')}<br>
                    ✏️ <strong>Recommendation:</strong> {issue.get('recommendation', '')}
                </div>""", unsafe_allow_html=True)
        else:
            st.success("No medium-severity issues found!")

    with tab4:
        items = result.get("compliant_items", [])
        if items:
            for item in items:
                st.markdown(f"<div class='compliant-item'>✅ {item}</div>", unsafe_allow_html=True)
        else:
            st.warning("No clearly compliant items identified.")

    # Download report
    st.divider()
    report = f"""MARKETPLACE COMPLIANCE REPORT
Platform: {result.get('platform', platform)}
Risk Score: {result.get('risk_score', 'N/A')}/100
Risk Level: {result.get('risk_level', 'N/A')}

SUMMARY:
{result.get('summary', '')}

RECOMMENDATION:
{result.get('overall_recommendation', '')}

CRITICAL ISSUES ({len(result.get('critical_issues', []))}):
{chr(10).join(f"- {i['clause']}: {i['issue']}" for i in result.get('critical_issues', []))}

HIGH ISSUES ({len(result.get('high_issues', []))}):
{chr(10).join(f"- {i['clause']}: {i['issue']}" for i in result.get('high_issues', []))}

COMPLIANT ITEMS:
{chr(10).join(f"- {i}" for i in result.get('compliant_items', []))}
"""
    st.download_button(
        "📥 Download Full Report",
        data=report,
        file_name="compliance_report.txt",
        mime="text/plain",
        use_container_width=True
    )
