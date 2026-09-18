import os
import tempfile

import streamlit as st

from rag import process_pdf, ask_question


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Research Paper RAG",
    page_icon="📄",
    layout="wide"
)


# ============================================================
# SESSION STATE
# ============================================================

if "rag_data" not in st.session_state:
    st.session_state.rag_data = None

if "file_name" not in st.session_state:
    st.session_state.file_name = None


# ============================================================
# TITLE
# ============================================================

st.title("📄 Research Paper Question Answering System")

st.write(
    "Upload a research paper and ask questions. "
    "The system retrieves relevant sections and generates "
    "an answer with page citations."
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("📚 Upload Research Paper")

    uploaded_file = st.file_uploader(
        "Choose a PDF",
        type=["pdf"]
    )

    if uploaded_file is not None:

        if st.session_state.file_name != uploaded_file.name:

            with st.spinner(
                "Processing research paper..."
            ):

                try:

                    with tempfile.NamedTemporaryFile(
                        delete=False,
                        suffix=".pdf"
                    ) as temp_file:

                        temp_file.write(
                            uploaded_file.getbuffer()
                        )

                        temp_path = temp_file.name

                    rag_data = process_pdf(temp_path)

                    os.unlink(temp_path)

                    st.session_state.rag_data = rag_data
                    st.session_state.file_name = uploaded_file.name

                    st.success("Paper processed successfully!")

                except Exception as e:

                    st.error(
                        f"Error processing PDF: {str(e)}"
                    )


# ============================================================
# MAIN AREA
# ============================================================

if st.session_state.rag_data is None:

    st.info(
        "👈 Upload a research paper from the sidebar to begin."
    )

else:

    rag_data = st.session_state.rag_data

    st.success(
        f"Loaded paper: {st.session_state.file_name}"
    )

    # --------------------------------------------------------
    # PAPER STATISTICS
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Pages",
            len(rag_data["pages"])
        )

    with col2:
        st.metric(
            "Text Chunks",
            len(rag_data["chunks"])
        )

    st.divider()

    # --------------------------------------------------------
    # QUICK QUESTIONS
    # --------------------------------------------------------

    st.subheader("⚡ Quick Questions")

    col1, col2, col3, col4 = st.columns(4)

    question = None

    with col1:
        if st.button("🎯 Objective"):
            question = "What is the main objective of this research paper?"

    with col2:
        if st.button("⚙️ Methodology"):
            question = "What methodology is used in this research paper?"

    with col3:
        if st.button("📊 Findings"):
            question = "What are the main findings of this research paper?"

    with col4:
        if st.button("⚠️ Limitations"):
            question = "What are the limitations of this research paper?"

    # --------------------------------------------------------
    # QUESTION INPUT
    # --------------------------------------------------------

    user_question = st.text_input(
        "Ask a question about the research paper:",
        placeholder="Example: What dataset was used?"
    )

    if st.button("🔍 Ask Question", type="primary"):

        if not user_question.strip():

            st.warning("Please enter a question.")

        else:

            question = user_question

    # --------------------------------------------------------
    # ANSWER
    # --------------------------------------------------------

    if question:

        with st.spinner("Searching the research paper..."):

            try:

                answer, retrieved_chunks = ask_question(
                    question,
                    rag_data
                )

                st.subheader("💡 Answer")

                st.write(answer)

                # ------------------------------------------------
                # SOURCES
                # ------------------------------------------------

                st.subheader("📖 Sources")

                for i, chunk in enumerate(
                    retrieved_chunks,
                    start=1
                ):

                    with st.expander(
                        f"Source {i} — Page {chunk['page']}"
                    ):

                        st.write(chunk["text"])

            except Exception as e:

                st.error(
                    f"Error generating answer: {str(e)}"
                )
