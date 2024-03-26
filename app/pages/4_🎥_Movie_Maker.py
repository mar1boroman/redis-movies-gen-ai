import streamlit as st
import redis
import os
import json
import random

from dotenv import load_dotenv
from redisvl.index import SearchIndex
from redisvl.utils.vectorize import OpenAITextVectorizer
from redisvl.query import VectorQuery
from openai import OpenAI
from rich import print

load_dotenv(dotenv_path="app.config")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def get_redis_conn(decode_responses=True) -> redis.Redis:
    redis_host, redis_port, redis_user, redis_pass = (
        os.getenv("redis_host"),
        os.getenv("redis_port"),
        os.getenv("redis_user"),
        os.getenv("redis_pass"),
    )
    if not redis_user:
        r = redis.Redis(
            host=redis_host, port=redis_port, decode_responses=decode_responses
        )
    else:
        r = redis.Redis(
            host=redis_host,
            port=redis_port,
            username=redis_user,
            password=redis_pass,
            decode_responses=decode_responses,
        )
    return r


results = []

def generate_prompt(results):
    if results:
        with st.form(key="prompt_issued").expander(label="Prompt Issued"):
            descriptions = ""
            for i, doc in enumerate(results):
                descriptions += f"<br>Description {i+1}   <br>"
                descriptions += doc + "  <br>"

            co, cc = "{", "}"
            response = f"""
                You are a expert story teller.  
                You are given the below movie descriptions    
                {descriptions}   
                Your task is to generate a movie plot which combines the essence of   
                all the above plotlines and generate a completely new story.  
                
                
                The output should be a JSON document in the following format
                {co}
                    "original_title" : "title generated",
                    "overview" : "overview generated",
                    "genres" : [List of genres of the generated movie, all double quoted],
                    "tagline" : "tagline of the movie",
                    "poster_desc" : "poster description generated"
                {cc}
                
                The overview should be atleast 200 words.  
                The poster description generated should be a simple image description  
                which can be fed to AI Image generator for best results, no more than one line.
                """

            st.markdown(response, unsafe_allow_html=True)
            submitted = st.form_submit_button("Submit", disabled=True)
            return response


def update_output_and_save(augmented_prompt, save):

    if augmented_prompt:

        client = OpenAI(api_key=OPENAI_API_KEY)

        with st.chat_message("assistant"):
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": augmented_prompt}],
                temperature=0,
            )
            reply = response.choices[0].message.content

            payload = json.loads(reply)

            poster_image = client.images.generate(
                model="dall-e-3",
                prompt=payload["poster_desc"],
                size="1024x1024",
                quality="standard",
                n=1,
            )

            st.markdown(
                f"""
                <img src="{poster_image.data[0].url}" alt="drawing" width="512" height="512"/>
                
                <br>  
                
                
                # {payload['original_title']}   <br>
                
                **{payload['tagline']}**  
                *_{','.join(payload['genres'])}_*  <br>
                
                *{payload['overview']}*  <br>
                
                """,
                unsafe_allow_html=True,
            )

            if save:
                oai = OpenAITextVectorizer(
                    model="text-embedding-ada-002",
                    api_config={"api_key": OPENAI_API_KEY},
                )
                gen_movie_embedding = oai.embed(text=p1)
                gen_movie_id = random.randint(500000, 1000000)

                add_movie = {
                    "id": gen_movie_id,
                    "original_title": payload["original_title"],
                    "original_language": "en",
                    "overview": payload["overview"],
                    "genres": payload["genres"],
                    "popularity": 0.0,
                    "runtime": 120,
                    "tagline": payload["tagline"],
                    "budget": 0,
                    "revenue": 0,
                    "vote_count": 0,
                    "vote_average": 10.0,
                    "embedding": gen_movie_embedding,
                }

                r = get_redis_conn()
                r.json().set(f"movie:{gen_movie_id}", "$", obj=add_movie)
                print(
                    f"The movie with keyname movie:{gen_movie_id} with title {payload['original_title']} has been added to the database!"
                )


def generate_movie(results, save=False):
    augmented_prompt = generate_prompt(results)
    update_output_and_save(augmented_prompt, save)


st.title("💬 Make a Movie!")
st.caption("🚀 Add Plotlines and generate a completely new movie!")

with st.sidebar:
    with st.form("my_form"):
        p1 = st.text_area(
            "Plotline 1",
            value="Action packed movie where Ethan Hunt and team pulls off impossible missions",
        )
        p2 = st.text_area(
            "Plotline 2",
            value="A boy discovers he is a wizard on his 11th birthday and goes on to join a wizarding school Hogwarts",
        )
        p3 = st.text_area(
            "Plotline 3",
            value="Hindi movie, A group of friends do a memorable road trip in Spain, perform various adventure sports including Skydiving, Deep sea diving, etc",
        )
        save = st.toggle("Save the generated movie to the database?")
        submitted = st.form_submit_button("Submit")

        if submitted:

            r = get_redis_conn()
            oai = OpenAITextVectorizer(
                model="text-embedding-ada-002",
                api_config={"api_key": OPENAI_API_KEY},
            )

            jindex = SearchIndex.from_yaml("movie_index.yaml")
            jindex.set_client(client=r)

            results = []
            for embedding in (
                oai.embed(text=p1),
                oai.embed(text=p2),
                oai.embed(text=p3),
            ):
                q = VectorQuery(
                    embedding,
                    "embedding",
                    return_fields=["original_title", "overview"],
                    num_results=1,
                )

                result = jindex.query(q)
                title, desc = result[0]["original_title"], result[0]["overview"]
                results.append(desc)
                print(result)
                print(title)

generate_movie(results=results, save=save)
