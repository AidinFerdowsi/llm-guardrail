import streamlit as st
import langchain_helper as lch
import textwrap

st.title("Webpage Summarizer")

with st.sidebar:
    with st.form(key='my_form'):
        webpage_url = st.sidebar.text_area(
        label="Enter the webpage URL:",
        max_chars=200
        )
        query = st.sidebar.text_area(
            label="Ask me about the video?",
            max_chars=50,
            key="query"
            )
        # openai_api_key = st.sidebar.text_input(
        #     label="OpenAI API Key",
        #     key="langchain_search_api_key_openai",
        #     max_chars=50,
        #     type="password"
        #     )
        "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"
        "[View the source code](https://github.com/rishabkumar7/pets-name-langchain/tree/main)"
        submit_button = st.form_submit_button(label='Submit')

if query and webpage_url:
    # if not openai_api_key:
    #     st.info("Please add your OpenAI API key to continue.")
    #     st.stop()
    # else:
    db = lch.create_db_from_webpage_url(webpage_url)
    response, docs = lch.get_response_from_query(db, query)
    st.subheader("Answer:")
    st.text(textwrap.fill(response, width=85))