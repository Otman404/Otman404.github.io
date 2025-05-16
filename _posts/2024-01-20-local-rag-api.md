---
title: Building a Local RAG api with LlamaIndex, Qdrant, Ollama and FastAPI
date: 2024-02-01
published: true
categories: [project, llms, rag]
tags: [llm, rag, llamaindex, huggingface, ollama, fastapi]     # TAG names should always be lowercase
image:
  path: /assets/headers/research-rag.png
#   lqip: 
render_with_liquid: false
---

## Introduction

ChatGPT has changed the way we interact with AI. People now use these Large Language Models (LLMs) as primary personal assistants for writing, brainstorming, and even consulting. However, the problem with these LLMs is that they are only as good as the data they are trained on. For example, if a company wanted to query ChatGPT about an internal document, ChatGPT wouldn’t be able to understand it. Additionally, its knowledge may not be up to date, and it is prone to hallucination.

To address this issue, we can fine-tune these LLMs on our data so they can answer questions from it. However, this approach can be very expensive. Keeping them updated with new information requires constant retraining. Moreover, the problem of hallucination remains, and it’s difficult to verify the sources of their answers.

A better approach would be to leverage powerful LLMs with access to our data without fine-tuning. How can we achieve this? Thanks to a recently emerged approach called [Retreival Augmented Generation (RAG)](https://arxiv.org/abs/2005.11401), we can retrieve relevant documents related to the user query and feed them as additional context to the LLM to generate the answer. (More details will be discussed in the next sections.)

![RAG Architecture Original](/assets/local-llamaindex-rag/rag_original.png)
*Source: [Original Paper](https://arxiv.org/abs/2005.11401)*

## Motivation

There are tons of tutorials over the internet on how to build a RAG pipeline, the issue is, the majority rely on online services and cloud tools, especially for the generation part, many tutorials advocate for the use of OpenAI LLM APIs, which, unfortunately, are not always free and may not be deemed trustworthy, particularly when handling sensitive data.

That's why I tried to build an end-to-end RAG api entirely offline and completely free, which can be very useful for companies that don't want to send their sensitive data over to a cloud service or a blackbox online LLM Api.

## Project Overview

We'll create an API endpoint where users can ask questions. The endpoint will search through a collection of research papers to find the answer. Then, it will use a Large Language Model (LLM) to process the answer and return it in a simple and easy-to-understand format.

![Local Rag Architecture](/assets/local-llamaindex-rag/local-rag-architecture.png)
*Local RAG Pipeline Architecture*

The projects consists of 4 major parts:

- Building RAG Pipeline using [Llamaindex](https://llamaindex.ai/)
- Setting up a local [Qdrant](https://qdrant.com/) instance using Docker
- Downloading a quantized LLM from [hugging face](https://www.huggingface.co/) and running it as a server using [Ollama](https://ollama.ai/)
- Connecting all components and exposing an API endpoint using [FastApi](https://fastapi.tiangolo.com/)

Alright, let's start building the project.

![Building Blocks Gif](https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExZ2c0dGtxZTNobnA1emZhanQ0cDhuOWozZm92aGZkZGdmZHN2d2oycCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/vZFZFVYQvtdidWZltF/giphy.gif){: w="300" h="150" }

## Building the project

We'll follow the common flow of a RAG pipeline, which is a bit similar to a standard ETL piepline.
![Rag Flow](/assets/local-llamaindex-rag/rag-flow.png)

>Code for the project can be found in this [Github repository](https://github.com/Otman404/local-rag-llamaindex){:target="_blank"}. Don't forget to <i class="fa-solid fa-star" style="color: #FFD43B;"></i>
{: .prompt-tip}

### 1. Loading

First, we need to get some data, in our case, research papers.
Let's create a python script to quickly download papers from [arxiv.org](https://arxiv.org/) based on some search keywords.
Luckily, there's already a [python wrapper](https://pypi.org/project/arxiv/) for the arxiv api.

The following function is responsible for downloading research papers into a given path.

```python
def download_papers(self, search_query, download_path, max_results):
    self._create_data_folder(download_path)
    client = arxiv.Client()
    logger.info("Searching papers from arxiv...", query=search_query)
    search = arxiv.Search(
        query=search_query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
    )

    results = list(client.results(search))
    for i, paper in enumerate(results):
        if os.path.exists(download_path):
            paper_title = (paper.title).replace(" ", "_")
            paper.download_pdf(dirpath=download_path, filename=f"{paper_title}.pdf")
            logger.info(
                f"{i+1}/{len(results)} PDF Downloaded", pdf_name=paper.title
            )
```
### 2. Indexing / Storing

Indexing involves structuring and representing our data in a way that facilitates storage, querying, and feeding to an LLM. Llamaindex provides several methods for accomplishing this task.

#### Chuncking

Initially, we need to **chunk** our data using a specified `chunk_size`. This is necessary because documents can often be too lengthy, which can introduce noise and exceed the context length for our LLM.

#### Embedding

Next, we proceed to embed these chunks using an pre-trained model. You can learn more about them in my [previous article](https://www.otmaneboughaba.com/posts/artwork-similarity-search)

There are many available powerful pre-trained model to use for embeddings. For this project, we'll use
[sentence-transformers/all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2)
![sentence-transformers/all-mpnet-base-v2](/assets/local-llamaindex-rag/embedder.png)
*sentence-transformers/all-mpnet-base-v2 on Hugging Face*

#### Storing

For storing our data, we'll use a local Qdrant docker instance, which can be easily set up with:

```bash
docker run -p 6333:6333 qdrant/qdrant
```

However, since we require our data to persist on disk even after stopping the instance, we must add a volume.

```bash
docker run -p 6333:6333 -v ~/qdrant_storage:/qdrant/storage:z qdrant/qdrant
```

> `qdrant_storage` is the name of a folder that we created in our HOME directory where data from qdrant db instance will be saved.
{: .prompt-info}

Then we can access our database UI from <http://localhost:6333/dashboard>


Llamaindex supports variaty of [Data Loaders](https://llamahub.ai/), in our case, our data will be a bunch of PDFs in a folder, which means we can use [`SimpleDirectoryReader`](https://docs.llamaindex.ai/en/stable/examples/data_connectors/simple_directory_reader.html)

The following function is responsible for loading, chunking, embedding and storing our data.

```python
def ingest(self, embedder, llm):
    logger.info("Indexing data...")
    documents = SimpleDirectoryReader(data_path).load_data()

    client = qdrant_client.QdrantClient(url=qdrant_url)
    qdrant_vector_store = QdrantVectorStore(
        client=client, collection_name=collection_name
    )
    storage_context = StorageContext.from_defaults(vector_store=qdrant_vector_store)

    Settings.llm = None
    Settings.embed_model = embedder
    Settings.chunk_size = chunk_size
    index = VectorStoreIndex.from_documents(
        documents, storage_context=storage_context, Settings=Settings
    )
    logger.info(
        "Data indexed successfully to Qdrant",
        collection=collection_name,
    )
    return index
```

Let's ingest some research papers about LLMs into our Vector Database. We'll do that by running the python script we just created:

```python
uv run data/data.py --query "LLM" --max 5 --ingest
```

`--query`: search keyword for downloading research papers from Arxiv.

`--max`: Limits the results.

`--ingest`: Ingests the downloaded papers into the Qdrant database.


> The full code of `Data` class can be found here: [data.py](https://github.com/Otman404/local-rag-llamaindex/blob/master/data/data.py)
{: .prompt-tip}

We can see that our collection is now created

![Qdrant UI](/assets/local-llamaindex-rag/qdrant_ui.png)
*Qdrant dashboard UI*

### 3. Querying

Now that we've successfully loaded our data (research papers) into our vector store (Qdrant), we can begin querying it to retrieve relevant data to be used by our LLM.

Let's begin by crafting a function that sets up our Qdrant index, which will serve as our query engine.

#### Query Engine

```python
def qdrant_index(self):
    client = qdrant_client.QdrantClient(url=qdrant_url)
    qdrant_vector_store = QdrantVectorStore(
        client=client, collection_name=collection_name
    )
    Settings.llm = self.llm
    Settings.embed_model = self.load_embedder()
    Settings.chunk_size = chunk_size

    index = VectorStoreIndex.from_vector_store(
        vector_store=qdrant_vector_store, settings=Settings
    )
    return index
```

> Code from: [rag.py](https://github.com/Otman404/local-rag-llamaindex/blob/master/api/rag.py)
{: .prompt-tip}

#### LLM

The goal is to use a local LLM, which can be a bit challenging since powerfull LLMs can be resource heavy and expensive. But thanks to model quantization, and Ollama, the process can be very easy.

Please refer to my previous article to learn more about setting up a local LLM with Ollama: [Use custom LLMs from Hugging Face locally with Ollama](https://otmaneboughaba.com/posts/local-llm-ollama-huggingface/)

After downloading `zephyr-7b-beta.Q5_K_M.gguf` from Hugging Face, we need to create a Modelfile for Ollama.

```text
FROM models/zephyr-models/zephyr-7b-beta.Q5_K_M.gguf
PARAMETER num_ctx 3900
PARAMETER temperature 0.7
PARAMETER top_k 50
PARAMETER top_p 0.95
PARAMETER stop "<|system|>"
PARAMETER stop "<|user|>"
PARAMETER stop "<|assistant|>"
PARAMETER stop "</s>"
TEMPLATE """
<|system|>\n system
{{ .System }}</s>
<|user|>\n user
{{ .Prompt }}</s>
<|assistant|>\n 
"""

SYSTEM """
As a personal assistant for researchers, your task is to analyze the provided research papers and extract pertinent information on using the provided keywords. Summarize key findings, methodologies, and any notable insights. This assistant plays a crucial role in facilitating researchers' understanding of the current state of knowledge on the provided keywords.
Your Motive:
Give the learner "aha" moment on every Topic he needs to understand. You can do this with the art of explaining things.
Focus on Conciseness and Clarity: Ensure that the output is concise yet comprehensive. Focus on clarity and readability to provide researchers with easily digestible insights.
IMPORTANT:
If the user query cannot be answered using the provided context, do not improvise, you should only answer using the provided context from the research papers.
If the user asks something that does not exist within the provided context, Answer ONLY with: 'Sorry, the provided query is not clear enough for me to answer from the provided research papers'.
"""
```

The Modelfile basically contains the parameters used by the LLM and a system prompt to improve how the it will behave in our context.

Next, we create our model using the Modelfile.

```bash
ollama create research_assistant -f Modelfile
```

This command will make our model available to Ollama, you can check that by running:

```bash
ollama list
```
You should see the model we just created.

Then, we start the model server:

```bash
ollama run research_assistant
```

By default, Ollama runs on `http://localhost:11434`

Finally, we create an API endpoint using FastAPI. This endpoint will receive a query, search the documents, and return a response.

An advantage of using FastAPI is its compatibility with Pydantic, which is very helpful in structuring our code and API responses.

Let's begin by defining two models: one for the Query and one for the Response:

```python
class Query(BaseModel):
    query: str
    similarity_top_k: Optional[int] = Field(default=1, ge=1, le=5)


class Response(BaseModel):
    search_result: str 
    source: str
```

After initiating the llm, qdrant index and our FastAPI app:

```python
llm = Ollama(model=llm_name, base_url=llm_url)
rag = RAG(llm=llm)
index = rag.qdrant_index()
```

We'll create a route that receives a `Query` and returns a `Response`, as defined in our pydantic classes.

```python
prompt = "You can only answer based on the provided context. If a response cannot be formed strictly using the context, politely say you don’t have knowledge about that topic"

@app.post("/api/search", response_model=Response, status_code=200)
def search(query: Query):

    query_engine = index.as_query_engine(
        similarity_top_k=query.similarity_top_k,
        output=Response,
        response_mode="tree_summarize",
        verbose=True,
    )
    response = query_engine.query(query.query + prompt)
    response_object = Response(
        search_result=str(response).strip(),
        source=[response.metadata[k]["file_path"] for k in response.metadata.keys()][0],
    )
    return response_object
```

> Code from: [main.py](https://github.com/Otman404/local-rag-llamaindex/blob/master/api/main.py)
{: .prompt-tip}

Here's the `.env` file containing the parameters used in the project:

```properties
DATA_PATH=papers/
LLM_URL=http://localhost:11434
LLM_NAME=research_assistant
EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2
QDRANT_URL=http://localhost:6333
COLLECTION_NAME=researchpapers
CHUNK_SIZE=1024
```


Let's run the fastapi server:

```bash
fastapi run api/main.py
```

And try the api with the following request:

```json
{
  "query": "How can robots imitate human actions?",
  "similarity_top_k": 3
}
```

For testing purposes, we'll just use the docs UI that comes with FastAPI out of the box

![API Request](/assets//local-llamaindex-rag/post_request.png)
*API Request*

Response
![API Response](/assets//local-llamaindex-rag/response.png)
*API Response*

Great! The api was able to retreive relevant context from our documents to return a well structured answer alongside citing the sources.

## Conclusion

In summary, the project's goal was to create a local RAG API using LlamaIndex, Qdrant, Ollama, and FastAPI. This approach offers privacy and control over data, especially valuable for organizations handling sensitive information.

>Don't forget to visit the project [Github repository](https://github.com/Otman404/local-rag-llamaindex)
{: .prompt-tip}

## What's next

In the [next article](https://otmaneboughaba.com/posts/dockerize-rag-application/), we'll learn how to Dockerize our RAG application step by step. We'll containerize each component: rag api, the data ingestion tool, the Qdrant vector database, and the Ollama LLM runtime, and orchestrate them all using Docker Compose.