import pandas as pd
import redis
from rich import print
import ast
from redisvl.index import SearchIndex
import time
from dotenv import load_dotenv
import os

load_dotenv(dotenv_path="app.config")


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


def dataframe_to_json(df: pd.DataFrame):
    records = df.to_dict(orient="records")
    return [dict(record) for record in records]


def create_index(r, indexfile, datafile):
    # Clean existing data
    r.flushdb()
    print("Existing data deleted from redis")

    # Preprocess data
    print("Data preprocessing started")
    df = pd.read_csv(datafile).fillna(value="")
    df["genres"] = df["genres"].apply(ast.literal_eval)
    df["embedding"] = df["embedding"].apply(ast.literal_eval)
    list_docs = dataframe_to_json(df)
    print("Data preprocessing complete")

    # construct a search index from the yaml file
    jindex = SearchIndex.from_yaml(indexfile)
    jindex.set_client(client=r)
    jindex.create(overwrite=True)
    print("Index created")
    jindex.load(list_docs, id_field="id")
    print("Data Loaded")

    print(jindex.info())


def main():
    start_time = time.time()
    filename = "utils/data_with_embeddings.csv"
    indexfile = "movie_index.yaml"
    create_index(r=get_redis_conn(), indexfile=indexfile, datafile=filename)
    print(f"Total time taken : {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
