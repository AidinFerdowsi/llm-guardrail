from bs4 import BeautifulSoup
import requests
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()
embeddings = OpenAIEmbeddings()

def fetch_webpage_content(url: str) -> str:
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    return ' '.join(soup.stripped_strings)

def create_db_from_webpage_url(webpage_url: str) -> FAISS:
    content = fetch_webpage_content(webpage_url)

    # Create a Document object with the webpage content
    doc = Document(page_content=content, metadata={"url": webpage_url})

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    
    # Now pass a list of Document objects (in this case, just one)
    docs = text_splitter.split_documents([doc])

    db = FAISS.from_documents(docs, embeddings)
    return db



def get_response_from_query(db, query, k=4):
    """
    gpt-3.5-turbo can handle up to 4097 tokens. Setting the chunksize to 1000 and k to 4 maximizes
    the number of tokens to analyze.
    """

    docs = db.similarity_search(query, k=k)
    docs_page_content = " ".join([d.page_content for d in docs])

    llm = ChatOpenAI(model_name="gpt-3.5-turbo")

    prompt = PromptTemplate(
        input_variables=["question", "docs"],
        template="""
        You are a helpful assistant that that can answer questions about webpages based on
        the content provided for you from the webpage.
        
        Answer the following question in most concise way: {question}
        By searching the following webpage: {docs}
        
        Only use the factual information from the transcript to answer the question.
        
        If you feel like you don't have enough information to answer the question, say "I don't know".
        
        Your answers should be verbose and detailed.
        """,
    )

    chain = LLMChain(llm=llm, prompt=prompt)

    response = chain.run(question=query, docs=docs_page_content)
    response = response.replace("\n", "")
    return response, docs