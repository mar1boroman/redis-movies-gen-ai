import streamlit as st
import redis
import os
from dotenv import load_dotenv

from redisvl.utils.vectorize import OpenAITextVectorizer
from redisvl.query import VectorQuery
from redisvl.query.filter import Tag, Num
from redisvl.index import SearchIndex


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


def update_output_area(query_issued="", results=[]):

    if query_issued:
        st.markdown("#### Query Issued", unsafe_allow_html=True)
        st.markdown(
            f"""<span style="word-wrap:break-word;">```{query_issued}```</span>""",
            unsafe_allow_html=True,
        )

        response = ""
        for doc in results:
            # print(doc)
            response += f"""
            ### {doc["original_title"]}
            **_{doc["tagline"] if doc["tagline"] else 'NA'}_**  
            *{doc["overview"]}*  
            *Genres : {doc['genres']}*  
            *Popularity : {doc['popularity']}*  
            *Runtime : {doc['runtime']}*  
            *Budget : {doc['budget']}*  
            *Revenue : {doc['revenue']}*  
            *Vote Count : {doc['vote_count']}*  
            *Vote Average : {doc['vote_average']}*  
            """
        st.markdown(response)


st.title("💬 Recommend movies?")
st.caption("🚀 Recommend movies with similar descriptions and filtering metadata")

with st.sidebar:
    query_issued = ""
    results = []

    with st.form("my_form"):
        overview = st.text_area(
            "Movie Description",
            value="Action packed movie where Ethan Hunt and team pulls off impossible missions",
        )
        popularity_val = st.slider(
            "Popularity > ", min_value=0.0, max_value=550.0, value=0.0
        )
        runtime_val = st.slider(
            "Runtime < ", min_value=0.0, max_value=1260.0, value=0.0
        )
        budget_val = st.slider("Budget < ", min_value=0, max_value=380000000, value=0)
        revenue_val = st.slider(
            "Revenue > ", min_value=0, max_value=2787965087, value=0
        )
        vote_count_val = st.slider(
            "Vote Count > ", min_value=0, max_value=14075, value=0
        )
        vote_avg_val = st.slider(
            "Vote Average > ", min_value=0.0, max_value=10.0, value=0.0
        )
        genres = st.selectbox(
            "Select Genre",
            (
                "*",
                "action",
                "adventure",
                "animation",
                "comedy",
                "crime",
                "documentary",
                "drama",
                "family",
                "fantasy",
                "foreign",
                "history",
                "horror",
                "music",
                "mystery",
                "romance",
                "science fiction",
                "thriller",
                "tv movie",
                "war",
                "western",
            ),
        )

        # Every form must have a submit button.
        submitted = st.form_submit_button("Submit")

        def make_filter(
            popularity_val=None,
            runtime_val=None,
            budget_val=None,
            revenue_val=None,
            vote_count_val=None,
            vote_avg_val=None,
            genres=None,
        ):
            flexible_filter = (
                (Num("popularity") >= popularity_val)
                & (Num("runtime") <= runtime_val)
                & (Num("budget") <= budget_val)
                & (Num("revenue") <= revenue_val)
                & (Num("vote_count") >= vote_count_val)
                & (Num("vote_average") >= vote_avg_val)
                & (Tag("genres") == genres)
            )
            return flexible_filter

        if submitted:
            load_dotenv(dotenv_path="app.config")
            r = get_redis_conn()
            oai = OpenAITextVectorizer(
                model="text-embedding-ada-002",
                api_config={"api_key": os.getenv("OPENAI_API_KEY")},
            )
            overview_clean = (
                overview
                if overview != ""
                else "Action packed movie where Ethan Hunt and team pulls off impossible missions"
            )
            overview_embedding = oai.embed(text=overview_clean)
            jindex = SearchIndex.from_yaml("movie_index.yaml")
            jindex.set_client(client=r)
            query_issued = VectorQuery(
                overview_embedding,
                "embedding",
                return_fields=[
                    "id",
                    "original_title",
                    "original_language",
                    "overview",
                    "genres",
                    "popularity",
                    "runtime",
                    "tagline",
                    "budget",
                    "revenue",
                    "vote_count",
                    "vote_average",
                    "embedding",
                ],
                filter_expression=make_filter(
                    popularity_val=popularity_val,
                    runtime_val=runtime_val,
                    budget_val=budget_val,
                    revenue_val=revenue_val,
                    vote_count_val=vote_count_val,
                    vote_avg_val=vote_avg_val,
                    genres=genres if genres != "*" else None,
                ),
                num_results=3,
            )

            results = jindex.query(query_issued)

update_output_area(query_issued=query_issued, results=results)
