import streamlit as st
import redis
import os
from redisvl.query import VectorQuery
from dotenv import load_dotenv
from redisvl.index import SearchIndex
from redisvl.utils.vectorize import OpenAITextVectorizer
from rich import print
import time


# Build Redis Connection
def get_redis_conn() -> redis.Redis:
    redis_host, redis_port, redis_user, redis_pass = (
        os.getenv("redis_host"),
        os.getenv("redis_port"),
        os.getenv("redis_user"),
        os.getenv("redis_pass"),
    )
    if not redis_user:
        r = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
    else:
        r = redis.Redis(
            host=redis_host,
            port=redis_port,
            username=redis_user,
            password=redis_pass,
            decode_responses=True,
        )
    return r


# Response processing
def get_response(prompt):
    # Environment Configurations
    load_dotenv(dotenv_path="app.config")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    # Redis Configurations
    r = get_redis_conn()
    jindex = SearchIndex.from_yaml("movie_index.yaml")
    jindex.set_client(client=r)
    oai = OpenAITextVectorizer(
        model="text-embedding-ada-002",
        api_config={"api_key": OPENAI_API_KEY},
    )
    response = ""
    prompt_vector_embedding = oai.embed(text=prompt)
    q = VectorQuery(
        vector=prompt_vector_embedding,
        vector_field_name="embedding",
        return_fields=["original_title", "overview", "tagline"],
        num_results=3,
    )
    print(q)

    for doc in jindex.query(q):
        print(doc)
        response += f"""
        ### {doc["original_title"]}
        **_{doc["tagline"] if doc["tagline"] else 'NA'}_**  
        *{doc["overview"]}*
        """

    return response


# Streamlit app
st.title("💬 Which Movie was that?")
st.caption("🚀 A streamlit chatbot powered by OpenAI LLM")
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {
            "role": "assistant",
            "content": "Describe the movie, I ll get you the details!",
        }
    ]

# Load History
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).markdown(msg["content"], unsafe_allow_html=True)

# Chat Q & A
if prompt := st.chat_input():
    start_time = time.time()
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    response = get_response(prompt=prompt)
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.chat_message("assistant").markdown(response)
    st.chat_message("assistant").markdown(f"Response time: {time.time() - start_time}")
