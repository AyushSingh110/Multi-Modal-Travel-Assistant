# Multi-Modal Travel Assistant

Ask about a city. You get a short write-up, a set of photos, and a weather chart.

The app decides for itself where the facts come from: its own small database, or
a web search. It asks for the weather and the photos at the same time, so the
answer comes faster. It remembers the conversation, so "what about next week?"
only fetches the new weather.

Every outside service has a real version and a fake one, so **the whole app runs
without any API key**.

```bash
conda env create -f environment.yml && conda activate travel-agent
python scripts/seed_vectorstore.py
streamlit run src/travel_agent/ui/app.py
```

Full setup is in [RUN_COMMANDS.md](RUN_COMMANDS.md).

**Model: Groq is the default.** OpenAI and Anthropic are both built and tested
too - switching is one line in `.env` (`LLM_PROVIDER=anthropic`). Groq is the
default because the free allowance on the other two does not cover a day of
building plus a live demo, and a demo that dies on a quota error is the worst
outcome. Groq returns a real `tool_calls` payload, so the hand-written tool
runner is still tested against the real thing. With no keys at all, a mock model
takes over and sends real `tool_calls` payloads of its own.

---

## The interface

![The assistant answering a question about Tokyo](docs/screenshots/02-in-store-city.png)

The **agent trace** panel on the right is the important part. It shows what the
app decided and the numbers behind each choice.

| | |
|---|---|
| ![Routing decision](docs/screenshots/04-trace-routing.png) | ![Parallel measurement](docs/screenshots/03-trace-parallelism.png) |
| **Routing** - the score, the limit, and how close every city is | **Parallelism** - time if done one by one, against real time |
| ![Follow-up skipping](docs/screenshots/05-follow-up-skipped.png) | ![Graceful degradation](docs/screenshots/07-weather-api-broken.png) |
| **Memory** - what a follow-up skipped, and what that saved | **Degradation** - weather broken on purpose; the page still works |

---

## The graph

![The LangGraph topology](graph.png)

Boxes are jobs, arrows say what runs next. Dashed arrows are choices made while
the app runs. The picture is generated from the real graph by
`scripts/export_graph.py`.

| Box | What it does |
|---|---|
| `normalize_input` | Cleans the question, starts a fresh turn. |
| `classify_intent` | Finds the city and the dates, decides what kind of question this is. |
| `plan_tools` | Picks the fact source, then asks the model which tools to call. |
| `retrieve_vector` | Reads the facts we stored. A database read, not a tool call. |
| `web_search` | Searches the web, for a city we did not store. |
| `execute_weather` | Runs the weather tool. Hand-written runner. |
| `execute_images` | Runs the photo tool. Same runner. |
| `join` | Waits for the parallel steps, measures the time saved. |
| `synthesize` | Builds the final checked answer. |

**Choice 1 - `route_after_intent`.** Is there any work to do? No city, or an
answer we already have, jumps straight to the end.

**Choice 2 - `route_and_fan_out`.** Picks the fact source (stored **or** web),
then returns a *list* of box names instead of one. Returning a list is what makes
the facts, weather and photo steps start together.

---

## Requirements

| Requirement | Where |
|---|---|
| LangGraph, clear nodes/edges/state | [`graph/builder.py`](src/travel_agent/graph/builder.py), [`graph/edges.py`](src/travel_agent/graph/edges.py) - 9 boxes, 2 choice points |
| Typed state | [`schemas/state.py`](src/travel_agent/schemas/state.py) - every shared value has a rule for joining |
| Streamlit GUI | [`ui/app.py`](src/travel_agent/ui/app.py), [`ui/components/`](src/travel_agent/ui/components/) |
| OpenAI or Anthropic model | [`services/llm/openai.py`](src/travel_agent/services/llm/openai.py), [`services/llm/anthropic.py`](src/travel_agent/services/llm/anthropic.py) - both built |
| Vector store, 3 cities | [`data/city_facts/`](data/city_facts/), [`scripts/seed_vectorstore.py`](scripts/seed_vectorstore.py) - 27 pieces, FAISS with a NumPy backup |
| Web search fallback | [`tools/search/`](src/travel_agent/tools/search/) - mock, DuckDuckGo, Tavily |
| **Conditional edge on knowledge availability** | [`services/router.py`](src/travel_agent/services/router.py) - name list first, then word closeness against 0.07 |
| Structured output | [`schemas/response.py`](src/travel_agent/schemas/response.py) - `city_summary`, `weather_forecast`, `image_urls` |
| Line chart from that object | [`ui/components/charts.py`](src/travel_agent/ui/components/charts.py) - Plotly, shaded high/low band |
| Weather, 5-7 days | [`tools/weather/`](src/travel_agent/tools/weather/) - mock, or OpenWeatherMap live |
| Images | [`tools/images/`](src/travel_agent/tools/images/) - Wikimedia photos, or Unsplash live |
| Graceful errors | [`tools/registry.py`](src/travel_agent/tools/registry.py), [`tools/retry.py`](src/travel_agent/tools/retry.py) - one broken tool never breaks the page |
| `graph.png` | [`graph.png`](graph.png), [`scripts/export_graph.py`](scripts/export_graph.py) |

**327 tests, all passing with no API keys** (`pytest -q`).

---

## The three distinctions

### 1. Manual tool execution

[`tool_executor.py`](src/travel_agent/graph/nodes/tool_executor.py) reads
`state["messages"][-1].tool_calls` itself, checks the arguments, runs the tool,
and replies with a `ToolMessage` carrying the same id. Wrong id, and the model
gets confused. Calls run together, so one broken tool does not stop the others.
A failed tool still gets a reply, marked as an error; an unknown tool name gets
back the list of tools that do exist.

No ready-made helpers, and a test proves it:
`test_no_prebuilt_tool_calling_helpers_anywhere_in_the_source` reads every file
in `src/` and `scripts/` and fails if `ToolNode` or the prebuilt agent builders
appear in code. It skips comments, so we can still explain the choice.

### 2. Parallel fan-out

One choice point returns a list of box names, so three steps start together. It
is built into the shape of the graph, not hidden inside a node - which is why you
can see it in `graph.png`.

| Query | Route | One by one | Real | Faster |
|---|---|---|---|---|
| Tokyo | vector store | 1964 ms | 1157 ms | 1.70x |
| Paris | vector store | 2056 ms | 1276 ms | 1.61x |
| Kyoto | web search | 2865 ms | 1035 ms | 2.77x |

Average **2.03x**, about 1.1 seconds saved. You wait for the slowest step, not
for all of them added up - so on the stored path the real race is weather against
photos, and that sets the limit.

### 3. Memory and partial update

State is saved per conversation (`MemorySaver`, or `AsyncSqliteSaver` with
`CHECKPOINTER=sqlite`), keyed by `thread_id`. Ask about Tokyo, then ask "what
about next week?": the city comes from saved state, the dates move, and **only
the weather step runs again**.

```
TURN 2  "what about next week?"    intent=weather_only
        ran     : classify_intent, plan_tools, execute_weather, synthesize
        skipped : retrieve_vector, execute_images
        forecast: 2026-08-19 -> 2026-08-26   (the dates really moved)
        images  : kept from turn 1
```

Honestly measured: this saves **296 ms** of waiting but avoids **1184 ms** of
outside work. The two differ because the skipped steps used to run *alongside*
the weather step. The real saving is a photo request and a database read we would
have thrown away, plus their quota. **Running steps together saves time; skipping
saves money** - and money is what grows with users.

The skip is checkable: `timings` is cleared each turn, so what is left is a
record of what really ran, and a test asserts `execute_images` is missing on turn
two.

---

## Main decisions

| Decision | Why |
|---|---|
| LangGraph, not a chain | Choosing a path, running steps together, and re-running only a part are all about shape. A chain hides them in normal code. |
| Name list first, then word closeness | "Do we have this city?" is mostly a question about names. "NYC" would fail a closeness test outright. |
| Limit 0.07 | Measured, not guessed. Stored cities score 0.102-0.207, unknown ones 0.000-0.040, so 0.07 sits in the gap. The seeding script prints the table. |
| Simple word-count embeddings | The question is only "is this one of our three cities?". The alternative needs hundreds of megabytes or an API key. |
| FAISS with a NumPy backup | FAISS suits data that grows, but a failed install must not stop the app. |
| Ask for JSON and repair it | Works the same on all four models, and the error handling stays in code we can explain. |
| The model writes words only | Forecasts and photo links come from tools with a fixed shape. Asking the model to repeat them only invites changes. |
| Fake services by default | A demo that depends on someone else's API key can fail in the room. |

**What this project taught me.** The suite reached 293 passing tests before the
first live request - and that request came back with no photos. The live model
was offered both tools and called only the weather one. No test caught it,
because my mock always asked for both. Fake services test your code against your
own guesses, not against reality. The fix was not a better prompt: the screen
always needs all three parts, so the graph now fills in any tool the model forgot
and records it as `tools_added_by_graph`.

---

## Demo

| # | Type this | Look for |
|---|---|---|
| 1 | `Tell me about Tokyo` | **Internal knowledge base**, score 0.207 against the 0.07 limit |
| 2 | `what about next week?` | City carries over; `retrieve_vector, execute_images` **skipped**; dates move, photos do not |
| 3 | `Tell me about Kyoto` | **Live web search** - 0.040, under the limit; links in the write-up |
| 4 | Toggle **Break the weather API**, then `Tell me about Paris` | Write-up and photos still appear; a banner explains the failure |

---

## Layout

```
src/travel_agent/
  config/     typed settings; nothing else reads the environment
  schemas/    state, tool arguments, responses, trace events
  services/   embeddings, vector stores, retrieval, router, LLM drivers
  tools/      weather, images, search - each with a live and a mock version
  graph/      nodes, edges, builder, checkpointer, diagram export
  ui/         Streamlit app, the async bridge, render components
data/         seed corpus, offline fallback images
evals/        labelled queries for router accuracy
scripts/      seeding, graph export, smoke tests
tests/        327 tests, no API keys needed
```

## Live providers

Each one switches on its own:

```bash
LLM_PROVIDER=groq          GROQ_API_KEY=...
WEATHER_PROVIDER=live      OPENWEATHER_API_KEY=...
IMAGE_PROVIDER=live        UNSPLASH_ACCESS_KEY=...
SEARCH_PROVIDER=live       # DuckDuckGo needs no key; Tavily uses TAVILY_API_KEY
```

Check one end to end with `python scripts/live_smoke_test.py`.

## Attribution

Gallery photos come from Wikimedia Commons, credited with photographer and
licence in [`data/images/ATTRIBUTION.md`](data/images/ATTRIBUTION.md), read from
the Commons API rather than guessed. The `*.png` files in that folder are our own
placeholder images, used when Commons cannot be reached.
