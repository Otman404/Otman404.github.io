---
title: Model Context Protocol
date: 2025-04-13
published: true
categories: [tutorial]
tags: [llms, modelcontextprotocol, mcp]     # TAG names should always be lowercase
image:
  path: assets/headers/mcp.png
#   lqip: 
render_with_liquid: false
---

## Introduction

LLMs are advancing at a rapid pace lately. Each week, we witness breakthroughs with new, very powerful models. However, they all share a common limitation: the inability to execute tasks or perform actions independently. This raises the question: wouldn't it be nice if these LLMs could perform actions for us? For example, sending emails, checking if a folder is available in our Google Drive, reviewing our calendar, booking flights.. you get the point. 
Fortunately, this has become possible with techniques such as tools calling.


With tool calling for LLMs, it is now possible for them to perform actions and tasks they couldn't handle independently. By defining functions and providing descriptions for these tools, LLMs can determine, based on the conversation, when to use a tool and which arguments to provide.

So far so good! But one of the problems with this approach is that each LLM uses tools differently, making it time-consuming to integrate various tools with different LLMs. And this is precisely the issue that MCP solves.


## Model Context Protocol

Model Context Protocol (MCP) is a method introduced by [Anthropic](https://www.anthropic.com/news/model-context-protocol) as a way to standardize how LLMs can interact and use different tools and data sources.
It is a two way secure client-server connection between data/tools and ai applications.

![mcp-architecture](assets/mcp/mcp_architecture.png)
_MCP Architecture_

### MCP Components
**- Hosts:** LLM applications like Claude Desktop, Cursor, 5ire…

**- Clients:** Responsible for maintaining a 1:1 client-server connection. Learn more here: [Client Development Example](https://modelcontextprotocol.io/quickstart/client)

**- MCP Server:** The program that provides resources or specific capabilities to our LLM as context through the Model Context Protocol standard.

In order for the Host and the servers to communicate, the protocol uses [JSON-RPC](https://www.jsonrpc.org/), via either `stdio transport` for local processes, or `HTTP with SSE transport` for APIs and POST requests.

## Building a finance MCP server

Now that we understand what Model Context Protocol is and its architecture, let's build one.

We are going to build an MCP server that allows our LLM to retrieve up-to-date prices and news about stocks and cryptocurrencies.



### Project Requirements

In this project we'll use `uv` as our package manager, so make sure you have that [installed](https://docs.astral.sh/uv/getting-started/installation/)
We'll also need to install [yfinance](https://yfinance-python.org/) to get data about stock tickers, and of course, [mcp[cli]](https://github.com/modelcontextprotocol/python-sdk) to create the tools and run the server.

```bash
# creating uv project
uv init mcp-finance-server
cd mcp-finance-server

# creating virtual environment for the project
uv venv
source .venv/bin/activate

# installing yfinance
uv add yfinance

# installing mcp[cli]
uv add "mcp[cli]"
```


### Creating the mcp server

in `server.py`, we'll initiate our mcp server, add the tools and resources that we want our client to have access to, and then run the server.


```python
from mcp.server.fastmcp import FastMCP
from datetime import datetime

mcp = FastMCP("finance")
```
This will start a server called "finance"

Then we'll create the tools:

- `get_price_tool`: Get the price of a stock/cryptocurrency ticker
  - Arguments:
    - `ticker` (string): Required - Ticker name or alias (e.g., "BTC-USD", "AAPL")
    - `period` (string): Optional - Time period (e.g., "1d", "5d", "1mo"). Defaults to "1d"


```python
@mcp.tool(name="get_price", description="Get the price of a stock/crypto ticker.")
async def get_price_tool(ticker: str, period: str = "1d") -> str:

    ticker = ticker_mapper(ticker)
    try:
        response = await get_price(ticker, period)
        response.index = response.index.strftime("%B %d, %Y %I:%M")
        text = ""
        for index, row in response.iterrows():
            text += f"Open: ${row['Open']:.3f}, High: ${row['High']:.3f}, Low: ${row['Low']:.3f}, Close: ${row['Close']:.3f}, Volume: ${row['Volume']:.3f}\n "
        return text
    except KeyError:
        return f"{ticker} symbol not found"
```
The code above fetches historical price data for a given stock or crypto ticker and formats the response as text so our LLM can easily interpret it.

- `get_news_tool`: Get the news of a stock/cryptocurrency ticker.
  - Required arguments:
    - `ticker` (string): Required
    - `count` (string): Optional - Number of articles to retrieve (default: 5)

```python
@mcp.tool(name="get_news", description="Get the news of a stock/crypto ticker.")
async def get_news_tool(ticker: str, count: int = 5) -> str:

    ticker = ticker_mapper(ticker)
    try:
        response = await get_news(ticker, count)

        text = ""
        for news in response:
            publication_date = datetime.fromisoformat(
                news["content"]["pubDate"].replace("Z", "+00:00")
            ).strftime("%Y-%m-%d %H:%M:%S")
            article_link = news["content"]["provider"]["url"]
            article_title = news["content"]["title"]
            article_publisher = news["content"]["provider"]["displayName"]
            article_summary = news["content"]["summary"]
            text += f"Date: {publication_date}, Title: {article_title}, Publisher: {article_publisher}, Link: {article_link}, Summary: {article_summary}\n"
        return text
    except KeyError:
        return f"{ticker} symbol not found"
```
The code above retrieves recent news articles related to a given ticker and also formats the results to an easily readable text.

One of the issues I found while testing this server is that the LLM sometimes messes up the ticker for a certain stock or cryto, so to ensure it gets it correct, I have to add a [ticker_mapper](https://github.com/Otman404/finance-mcp-server/blob/be5305d7a89306738047879ac89e648d0cdb1e39/src/finance_mcp_server/server.py#L9C1-L135C2) so it corrects misspellings and make the ticker compatible with yfinance format:

```python
ticker_map = {
    # Cryptocurrencies (Mapping names and symbols to SYMBOL-USD format)
    "bitcoin": "BTC-USD",
    "btc usd": "BTC-USD",
    "btc-usd": "BTC-USD",
    "btc": "BTC-USD",
    "ethereum": "ETH-USD",
    "eth": "ETH-USD",
    "ripple": "XRP-USD",
    "xrp": "XRP-USD",
    "cardano": "ADA-USD",
    "ada": "ADA-USD",
    "solana": "SOL-USD",
    ...
}
```




and lastly, we run the server:
```python
def main() -> None:
    print("Starting Finance MCP server")
    mcp.run(transport="stdio")

if __name__ == "__main__":
    main()
```

From the command line, we can start the server with:

```bash
uv run server.py
```

### Adding the MCP server to the MCP client
Now that we've created and started the mcp server, it's time to add it to the client configuration.

For the client, we have many options to choose from. Take a look at this list for all the clients supporting MCP servers: [MCP Clients](https://modelcontextprotocol.io/clients)

For this demo, we'll use [Claude Desktop](https://claude.ai/download), so make sure you have it installed.


> I also recommend using [5ire](https://github.com/nanbingxyz/5ire), which is an open source client that supports Linux and also easy to use with local llms via Ollama.
{: .prompt-tip }

To add the mcp server, we need to update  `claude_desktop_config.json`, you can find it by clicking on **File** (top left)--> **Settings** --> **Developer** --> **Edit Config**, it'll open the directory that contains the json config file, open it and add:

```json
{
    "mcpServers": {
        "finance": {
            "command": "uv",
            "args": [
                "--directory",
                "/absolute/path/to/finance-mcp-server",
                "run",
                "server.py"
            ]
        }
    }
}
```

After adding our server to the client, the tools will be available for our LLM to use.

![Claude tools](assets/mcp/claude_tools.png)

So during a chat session, the llm will know when and which tool to use and the required arguments to provide, based on the tools descriptions.




### Let's try it

We'll ask claude to provide Apple stock price, and also some news about Bitcoin

![Server Test Example](assets/mcp/example.png)

Great! We can see that the MCP server successfully integrated with the client. The tools are now accessible, and the LLM can interact with them to fetch real-time data like stock prices and news.

### Running the server via docker

We can also run the server via Docker, which can become more convenient if we want to share the server with other users without dealing with the hassle of setting up the environment.

Let's first build the server image. Below in the dockerfile for this project.

> In order for the image to build correctly make sure you have the correct config in [pyproject.toml](https://github.com/Otman404/finance-mcp-server/blob/master/pyproject.toml). Notice `[project.scripts]` should point to the main function. Learn more: *[Writing pyproject.toml](https://packaging.python.org/en/latest/guides/writing-pyproject-toml/)*
{: .prompt-warning }

```Dockerfile
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS uv

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1

ENV UV_LINK_MODE=copy

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project --no-dev --no-editable

ADD . /app
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable

FROM python:3.10-slim

WORKDIR /app

COPY --from=uv /root/.local /root/.local
COPY --from=uv --chown=app:app /app/.venv /app/.venv

ENV PATH="/app/.venv/bin:$PATH"

ENTRYPOINT ["finance-mcp-server"]
```

```bash
# Build the image
docker build -t finance-server .

# run the container
docker run -it finance-server
```


In this case, we'll need to add this config to Claude Desktop config file to connect to the server:

```json
{
  "mcpServers": {
    "finance": {
      "command": "docker",
      "args": ["run", "-i", "--rm", "finance-server"]
    }
  }
}
```

And voila! We just built an mcp server 🚀 


## Final Notes

In this article, we covered: 

- What is Model Context Protocol
- Setting up the environment
- Creating an mcp server
- Adding two tools to the server with he help of `yfinance` library
- Running the server and integrating it with mcp client (Claude Desktop)
- Running the mcp server via `uv` and `docker`

I'll leave you with these resources the explore all the open source and ready to use mcp servers, just plug and play in your client. Enjoy 🎉 

- [modelcontextprotocol.io/examples](https://modelcontextprotocol.io/examples)
- [awesome-mcp-servers](https://github.com/punkpeye/awesome-mcp-servers)
- [smithery.ai/](https://smithery.ai/)
