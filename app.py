import streamlit as st
from main import initialize_system, ask_question

st.set_page_config(
    page_title="Denso Q&A",
    layout="centered"
)

if 'system_initialized' not in st.session_state:
    st.session_state.system_initialized = False

if not st.session_state.system_initialized:
    with st.spinner("Loading system..."):
        try:
            initialize_system()
            st.session_state.system_initialized = True
        except Exception as e:
            st.error(f"Error initializing system: {str(e)}")
            st.stop()

st.title("Denso Workflow Q&A")

question = st.text_input("Ask a question about the Denso workflow:", placeholder="e.g., What are the steps in TSB Validation?")

if st.button("Search", type="primary"):
    if question:
        with st.spinner("Searching..."):
            try:
                result = ask_question(question)
                st.write("**Answer:**")
                st.write(result['answer'])
            except Exception as e:
                st.error(f"Error: {str(e)}")
    else:
        st.warning("Please enter a question.")
