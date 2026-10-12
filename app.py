import streamlit as st

from src.core.rag_factory import (
    create_rag_pipeline,
)

# --------------------------------------------------
# Streamlit Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Enterprise RAG Assistant",
    page_icon="🤖",
    layout="wide",
)

# --------------------------------------------------
# Cache Pipeline
# --------------------------------------------------

@st.cache_resource
def get_pipeline():
    return create_rag_pipeline()

pipeline = get_pipeline()

# --------------------------------------------------
# Session State
# --------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

# --------------------------------------------------
# Header
# --------------------------------------------------

st.title("🤖 Enterprise RAG Assistant")
st.caption(
    "Ask questions about your uploaded company documents."
)

# --------------------------------------------------
# Display Chat History
# --------------------------------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(
            message["content"]
        )

# --------------------------------------------------
# User Input
# --------------------------------------------------

question = st.chat_input(
    "Ask a question..."
)

# --------------------------------------------------
# Process Question
# --------------------------------------------------

if question:

    # Show User Message

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    # Generate Response

    with st.chat_message("assistant"):

        with st.spinner(
            "Searching documents..."
        ):

            try:

                result = (
                    pipeline.query_with_sources(
                        question
                    )
                )

                answer = result["answer"]

                sources = result["sources"]

                st.markdown(answer)

                if sources:

                    with st.expander(
                        "📚 Sources"
                    ):

                        shown_sources = set()

                        for source in sources:

                            source_name = source.get(
                                "source",
                                "Unknown",
                            )

                            score = source.get(
                                "score",
                                0,
                            )

                            if (
                                source_name
                                not in shown_sources
                            ):

                                st.write(
                                    f"• {source_name} "
                                    f"(score: {score:.3f})"
                                )

                                shown_sources.add(
                                    source_name
                                )

                # Store Assistant Response

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                    }
                )

            except Exception as e:

                error_message = (
                    f"Error: {str(e)}"
                )

                st.error(error_message)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": error_message,
                    }
                )