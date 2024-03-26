# redis-movies-gen-ai
Redis Vector Similarity Search, Semantic Caching, Recommendation Systems and RAG

# About this demo application

This demo showcases different GenAI use cases with redis database
- Vector Search
- Semantic Caching
- Recommendation Systems
- RAG Framework for Gen AI


## Project Setup

### Spin up a Redis instance enabled with RedisStack!

The easiest way to is to use a docker image using the below command
```bash
docker run -d -p 6379:6379 -p 8001:8001 redis/redis-stack:latest
```

If you do not want to use a docker image, you can sign up for a free Redis Cloud subscription [here](https://redis.com/try-free).

###  Set up the project

Download the repository

```
git clone https://github.com/mar1boroman/redis-movies-gen-ai.git && cd redis-movies-gen-ai
```

Prepare and activate the virtual environment

```
python3 -m venv venv && source venv/bin/activate
```

Install necessary libraries and dependencies

```
pip install -r requirements.txt
```



### Using the project

#### Update Config

Make sure you update the app.config file. You need a open ai api key to update the config.
```
vi app.config
```

#### Load Data

Load the data into redis with embeddings and create index 
The data file is hosted in GCP bucket

```bash
curl -L -o utils/data_with_embeddings.csv.gz https://storage.googleapis.com/okon-datasets/data_with_embeddings.csv.gz
```

```bash
gunzip utils/data_with_embeddings.csv.gz
```

```python
python utils/load_redis.py 
```
#### Run application

Run the UI

```bash
streamlit run app/1_🔍_Find_My_Movies.py
```
