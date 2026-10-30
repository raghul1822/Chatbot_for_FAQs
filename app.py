"""
Streamlit Web User Interface for College Information FAQ Chatbot
CodeAlpha Artificial Intelligence Internship - Task 2
"""

import sys
from pathlib import Path
import streamlit as st

# Ensure project root is in system path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from chatbot import CollegeFAQChatbot

# Configure page settings
st.set_page_config(
    page_title="ApexTech College FAQ Chatbot",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.1rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.2rem;
    }
    .badge {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        font-size: 0.78rem;
        font-weight: 600;
        border-radius: 9999px;
        margin-right: 0.35rem;
    }
    .badge-blue { background-color: #DBEAFE; color: #1E40AF; }
    .badge-green { background-color: #D1FAE5; color: #065F46; }
    .badge-amber { background-color: #FEF3C7; color: #92400E; }
    .badge-purple { background-color: #EDE9FE; color: #5B21B6; }
    .faq-meta {
        background-color: #F8FAFC;
        border-left: 3px solid #2563EB;
        padding: 0.6rem 0.9rem;
        margin-top: 0.5rem;
        border-radius: 4px;
        font-size: 0.88rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading College Knowledgebase & NLP Models...")
def load_college_bot(dataset_path: str = "data/faqs.csv", threshold: float = 0.35) -> CollegeFAQChatbot:
    """Initializes and caches the College FAQ Chatbot."""
    return CollegeFAQChatbot(
        dataset_path=dataset_path,
        similarity_threshold=threshold,
        top_k=3,
        include_keywords=True,
    )


# Initialize Chatbot
bot = load_college_bot()

# --- SIDEBAR CONFIGURATION ---
with st.sidebar:
    st.image("https://img.icons8.com/clouds/200/university.png", width=105)
    st.title("🎓 College FAQ Controls")
    st.caption("CodeAlpha AI Internship · Task 2")

    st.markdown("---")
    st.subheader("⚙️ Matcher Settings")

    # Similarity threshold slider
    threshold_val = st.slider(
        "Similarity Threshold (Cutoff)",
        min_value=0.10,
        max_value=0.90,
        value=0.35,
        step=0.05,
        help="Queries scoring below this cosine similarity threshold will trigger a polite fallback response.",
    )
    bot.set_threshold(threshold_val)

    # Toggles
    show_suggestions = st.checkbox("Show Related FAQ Suggestions", value=True)
    debug_mode = st.checkbox("Enable Developer / Debug Mode", value=False)

    st.markdown("---")
    st.subheader("💡 Example Questions")
    quick_questions = [
        "How can I apply for undergraduate admission?",
        "What are the eligibility criteria for B.Tech?",
        "How much is the annual tuition fee?",
        "What are the central library operating hours?",
        "What amenities are provided in hostels?",
        "What is the average placement salary package?",
        "Can I order a pepperoni pizza with extra cheese?",  # Fallback test
    ]
    for q in quick_questions:
        if st.button(q, key=f"quick_{q}", use_container_width=True):
            st.session_state["pending_input"] = q

    st.markdown("---")
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    # Knowledgebase Browser
    st.markdown("---")
    with st.expander("📚 Browse College Knowledgebase (50 FAQs)"):
        categories = bot.get_categories()
        selected_cat = st.selectbox("Filter by Category", ["All"] + categories)
        faqs_to_show = (
            bot.records
            if selected_cat == "All"
            else bot.get_faqs_by_category(selected_cat)
        )
        st.write(f"Showing **{len(faqs_to_show)}** FAQs:")
        for f in faqs_to_show:
            with st.expander(f"Q: {f['question']}"):
                st.write(f"**Answer:** {f['answer']}")
                st.caption(f"ID: `{f['faq_id']}` | Category: `{f['category']}`")


# --- MAIN CHAT INTERFACE ---
col_head, col_tags = st.columns([3, 2])
with col_head:
    st.markdown('<div class="main-title">🎓 ApexTech College FAQ Chatbot</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Instant automated answers for Admissions, Academics, Hostels, and Placements</div>',
        unsafe_allow_html=True,
    )
with col_tags:
    st.markdown(
        """
        <div style="text-align: right; margin-top: 10px;">
            <span class="badge badge-blue">Python 3</span>
            <span class="badge badge-green">NLTK NLP</span>
            <span class="badge badge-purple">TF-IDF N-grams</span>
            <span class="badge badge-amber">Cosine Sim</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("---")

# Initialize conversation history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "👋 Welcome to the **ApexTech Institute of Technology FAQ Assistant**! "
                "Feel free to ask me anything about admissions, B.Tech/M.Tech courses, eligibility, "
                "fees, examination procedures, library hours, hostel rules, or placement training."
            ),
            "meta": None,
        }
    ]

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("meta"):
            meta = msg["meta"]
            if meta.get("status") == "matched" and debug_mode:
                st.markdown(
                    f"""
                    <div class="faq-meta">
                        🎯 <b>Matched FAQ:</b> <i>{meta['matched_question']}</i><br>
                        📊 <b>Cosine Similarity Score:</b> <code>{meta['confidence_score']:.4f}</code> (Cutoff: <code>{meta['threshold']:.2f}</code>) &nbsp;|&nbsp; 🏷️ <b>Category:</b> {meta['category']}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            elif meta.get("status") == "low_confidence_fallback" and debug_mode:
                st.markdown(
                    f"""
                    <div class="faq-meta" style="border-left-color: #F59E0B;">
                        ⚠️ <b>Low Confidence Match:</b> Top similarity score was <code>{meta['confidence_score']:.4f}</code> (below cutoff <code>{meta['threshold']:.2f}</code>).
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Show suggestions
            if meta.get("suggestions") and show_suggestions:
                with st.expander("🔍 Related FAQ Questions You Might Mean"):
                    for s in meta["suggestions"]:
                        st.markdown(f"- **{s['question']}** `(score: {s['score']:.2f})`")

            # Developer debug panel
            if meta.get("debug") and debug_mode:
                with st.expander("🛠️ Developer Debugging Analysis"):
                    st.json(meta["debug"])

# Check for pending prompt from sidebar button
pending_input = st.session_state.pop("pending_input", None)

# Accept input from chat input or quick prompt
user_query = st.chat_input("Ask a question about college admissions, fees, hostels, etc...") or pending_input

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query, "meta": None})
    with st.chat_message("user"):
        st.markdown(user_query)

    # Process query
    result = bot.ask(user_query, return_debug=debug_mode)

    meta = {
        "status": result["status"],
        "confidence_score": result["confidence_score"],
        "threshold": bot.similarity_threshold,
        "matched_question": result["matched_question"],
        "category": result["category"],
        "suggestions": [s for s in result["suggestions"] if s["question"] != result["matched_question"]],
        "debug": result.get("debug"),
    }

    with st.chat_message("assistant"):
        st.markdown(result["answer"])

        if result["status"] == "matched" and debug_mode:
            st.markdown(
                f"""
                <div class="faq-meta">
                    🎯 <b>Matched FAQ:</b> <i>{meta['matched_question']}</i><br>
                    📊 <b>Cosine Similarity Score:</b> <code>{meta['confidence_score']:.4f}</code> (Cutoff: <code>{meta['threshold']:.2f}</code>) &nbsp;|&nbsp; 🏷️ <b>Category:</b> {meta['category']}
                </div>
                """,
                unsafe_allow_html=True,
            )
        elif result["status"] == "low_confidence_fallback" and debug_mode:
            st.markdown(
                f"""
                <div class="faq-meta" style="border-left-color: #F59E0B;">
                    ⚠️ <b>Low Confidence Match:</b> Top similarity score was <code>{meta['confidence_score']:.4f}</code> (below cutoff <code>{meta['threshold']:.2f}</code>).
                </div>
                """,
                unsafe_allow_html=True,
            )

        if meta["suggestions"] and show_suggestions:
            with st.expander("🔍 Related FAQ Questions You Might Mean"):
                for s in meta["suggestions"]:
                    st.markdown(f"- **{s['question']}** `(score: {s['score']:.2f})`")

        if meta.get("debug") and debug_mode:
            with st.expander("🛠️ Developer Debugging Analysis"):
                st.json(meta["debug"])

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "meta": meta,
    })
