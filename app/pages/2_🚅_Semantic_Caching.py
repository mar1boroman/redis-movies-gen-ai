import time
import redis
import os
import rich

from redisvl.index import SearchIndex
from redisvl.query import VectorQuery
from redisvl.utils.vectorize import OpenAITextVectorizer
from redisvl.extensions.llmcache import SemanticCache

import streamlit as st
from rich import print
from openai import OpenAI
from dotenv import load_dotenv


# Build Redis Connection
def get_redis_conn(decode_responses=True) -> redis.Redis:
    redis_host, redis_port, redis_user, redis_pass = (
        os.getenv("redis_host"),
        os.getenv("redis_port"),
        os.getenv("redis_user"),
        os.getenv("redis_pass"),
    )
    if not redis_user:
        r = redis.Redis(host=redis_host, port=redis_port, decode_responses=decode_responses)
    else:
        r = redis.Redis(
            host=redis_host,
            port=redis_port,
            username=redis_user,
            password=redis_pass,
            decode_responses=decode_responses,
        )
    return r


def get_response(prompt):
    # Environment Configurations
    load_dotenv(dotenv_path="app.config")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    # Redis Configurations
    r = get_redis_conn()
    jindex = SearchIndex.from_yaml("movie_index.yaml")
    jindex.set_client(client=r)

    r_semantic = get_redis_conn(False)
    llmcache = SemanticCache(
        name="llmcache",
        prefix="llmcache",
        distance_threshold=0.2,
        redis_client=r_semantic,
    )

    # Try to get response from redis semantic cache
    def check_semantic_cache(prompt):
        if response := llmcache.check(prompt=prompt):
            print(f"Found similar doc in semantic cache")
            return response[0]["response"]
        else:
            return False

    if semantic_response := check_semantic_cache(prompt):
        print(semantic_response)
        return semantic_response
    else:
        response = ""
        oai = OpenAITextVectorizer(
            model="text-embedding-ada-002",
            api_config={"api_key": OPENAI_API_KEY},
        )
        prompt_vector_embedding = oai.embed(text=prompt)
        print(f"Query not found in semantic cache")

        q = VectorQuery(
            vector=prompt_vector_embedding,
            vector_field_name="embedding",
            return_fields=["original_title", "overview", "tagline"],
            num_results=1,
        )

        doc = jindex.query(q)[0]

        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        poster_image = client.images.generate(
            model="dall-e-3",
            prompt="Please make sure the image contains NO text at all."
            + doc["overview"],
            size="1024x1024",
            quality="standard",
            n=1,
        )
        response += f"""
        <img src="{poster_image.data[0].url}" alt="drawing" width="512" height="512"/>
            
        <br>  
        
        ### {doc["original_title"]}
        **_{doc["tagline"] if doc["tagline"] else 'NA'}_**  
        *{doc["overview"]}*
        """

        llmcache.store(prompt=prompt, response=response)
        return response


# Streamlit app
start_time = time.time()

st.title("💬 Poster Generator")
st.caption("🚀 Generate posters based on the movie descriptions")
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {
            "role": "assistant",
            "content": "Describe the movie, I ll get the details and generate a poster!",
        }
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).markdown(msg["content"], unsafe_allow_html=True)

if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    msg = get_response(prompt=prompt)
    st.session_state.messages.append({"role": "assistant", "content": msg})
    st.chat_message("assistant").markdown(msg, unsafe_allow_html=True)
    st.chat_message("assistant").markdown(f"Response time: {time.time() - start_time}")
