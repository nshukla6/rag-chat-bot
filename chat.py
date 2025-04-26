import os
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import openai
import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv


st.secrets.get()
# Load the .env file
load_dotenv(override=True)
# Now get the API key
openai_api_key = st.secrets.get("OPENAI_API_KEY")

openai.api_key = openai_api_key



st.title("I am your personal Assistant")
st.write("Hi! I am RAGA")


if "chat_history" not in st.session_state:
    st.session_state.chat_history=[]


for message in st.session_state.chat_history:
    with st.chat_message(message['role']):
        st.markdown(message['content'])


# Load prompt
prompt = hub.pull("jclemens24/rag-prompt")


# Load text file
@st.cache_data
def split_docs_file(file_name,chunk_size=400,chunk_overlap=100):
    with open(file_name, "r", encoding="utf-8") as f:
        text = f.read()

    # Create docs array
    docs = [Document(page_content=text, metadata={"source": file_name})]

    # Initialize splitter
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    # Split documents
    split_docs = splitter.split_documents(docs)
    print("execute split file")
    return split_docs

@st.cache_data
def split_docs_folder(folder_name,chunk_size=600,chunk_overlap=100, file_type="txt"):
    docs = []

    for file in os.listdir(folder_name):
        if file.endswith(F".{file_type}"):
            with open(os.path.join(folder_name, file), "r", encoding="utf-8") as f:
                text = f.read()
                docs.append(Document(page_content=text, metadata={"source": file}))
    
    # Initialize splitter
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    # Split documents
    split_docs = splitter.split_documents(docs)
    print("execute split folder")
    return split_docs

@st.cache_resource
def create_indexing(_split_docs=[]):
    vectorstore = Chroma.from_documents(
        documents = _split_docs,
        embedding = OpenAIEmbeddings(  
        )
    )
    print("execute vector indexing")
    return vectorstore.as_retriever()


def format_retrieved_docs(retrieved_docs):
    return "\n\n".join(doc.page_content for doc in retrieved_docs)


llm = ChatOpenAI(model_name="gpt-4o-mini",temperature=0)



split_docs = split_docs_file("jitendra.txt")


vectorstore_ref = create_indexing(split_docs)

# query = "who is doctor jitendra"

# relevant_docs = vectorstore_ref.get_relevant_documents(query)
# print(relevant_docs)


rag_chain = (
    {
        "context": vectorstore_ref|format_retrieved_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

# ans = rag_chain.invoke("what is the contact of doctor")

# print(ans)

prompt_input = st.chat_input("Whats UP")

def chat_with_rag(prompt_input):
    for output in rag_chain.stream(prompt_input):
        yield output




if prompt_input:
    st.session_state.chat_history.append({'role': 'user', 'content': prompt_input})
    with st.chat_message('user'):
        st.markdown(prompt_input)

    with st.chat_message('assistant'):
        response = st.write_stream(chat_with_rag(prompt_input))
    st.session_state.chat_history.append({'role': 'assistant', 'content': response})
