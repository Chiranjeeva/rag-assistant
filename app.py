import streamlit as st

from src.core.rag_factory import (
    create_rag_pipeline,
)

from src.security.user_context import (
    UserContext,
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
# Cached Pipeline
# --------------------------------------------------

@st.cache_resource
def get_pipeline():
    return create_rag_pipeline()

pipeline = get_pipeline()

# --------------------------------------------------
# User Role Selection
# --------------------------------------------------

st.sidebar.title("User Context")

role = st.sidebar.selectbox(
    "Role",
    [
        "employee",
        "manager",
        "hr",
        "admin",
    ]
)

user_context = UserContext(
    user_id="demo-user",
    role=role,
)


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
    f"Current role: {role}"
)

# --------------------------------------------------
# Show Chat History
# --------------------------------------------------

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):
        st.markdown(
            message["content"]
        )

# --------------------------------------------------
# Chat Input
# --------------------------------------------------

question = st.chat_input(
    "Ask a question..."
)

# --------------------------------------------------
# Answer
# --------------------------------------------------

if question:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):

        with st.spinner(
            "Searching documents..."
        ):

            try:

                result = (
                    pipeline.query_with_sources(
                        question=question,
                        user_context=user_context,
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

                            if source_name not in shown_sources:

                                st.write(
                                    f"• {source_name} "
                                    f"(score: {score:.3f})"
                                )

                                shown_sources.add(
                                    source_name
                                )

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                    }
                )

            except Exception as e:

                st.error(
                    f"Error: {str(e)}"
                )