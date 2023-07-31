import openai
import streamlit as st
import qdrant_client
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Qdrant
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI


def get_vector_store():

    client = qdrant_client.QdrantClient(
        st.secrets["QDRANT_HOST"],
        api_key=st.secrets["QDRANT_API_KEY"]
    )

    embeddings = OpenAIEmbeddings()

    vector_store = Qdrant(
        client=client,
        collection_name=st.secrets["QDRANT_COLLECTION_NAME"],
        embeddings=embeddings,
    )

    return vector_store


def create_answer_with_context(query):
    client = qdrant_client.QdrantClient(
        st.secrets["QDRANT_HOST"],
        api_key=st.secrets["QDRANT_API_KEY"]
    )

    response = openai.Embedding.create(
        input=query,
        model="text-embedding-ada-002"
    )
    embeddings = response['data'][0]['embedding']

    search_result = client.search(
        collection_name=st.secrets["QDRANT_COLLECTION_NAME"],
        query_vector=embeddings,
        limit=5
    )

    prompt = "Context:\n"
    for result in search_result:
        prompt += result.payload['text'] + "\n---\n"
    prompt += "You are Nicolas Motillon. Question:" + query

    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return completion.choices[0].message.content


st.title("nikolAI 🤖")

openai.api_key = st.secrets["OPENAI_API_KEY"]

# create vector store
vector_store = get_vector_store()

qa = RetrievalQA.from_chain_type(
    llm=OpenAI(),
    chain_type="stuff",
    retriever=vector_store.as_retriever()
)

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    response = create_answer_with_context(prompt)
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append(
        {"role": "assistant", "content": response})
