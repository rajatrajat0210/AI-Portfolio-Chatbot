import streamlit as st
from chains import Chain


def create_streamlit_app(llm):
    st.title("💼 Ask About Rajat's Resume")
    
    # Text input for user question
    question_input = st.text_input("Ask a question about Rajat's resume:")
    submit_button = st.button("Submit")

    if submit_button:
        try:
            # Validate if the user input question is provided
            if not question_input:
                st.warning("Please enter a question.")
                return

            # Process the question and generate a response using the 'answer_question' method
            response = llm.answer_question(question_input)
            st.subheader("Answer:")
            st.write(response)

        except Exception as e:
            st.error(f"An error occurred: {e}")


if __name__ == "__main__":
    # Initialize the Chain class (Groq-powered LLM)
    chain = Chain()

    # Set up Streamlit app configuration
    st.set_page_config(layout="wide", page_title="Ask About Rajat's Resume", page_icon="💼")
    
    # Start the Streamlit app
    create_streamlit_app(chain)
