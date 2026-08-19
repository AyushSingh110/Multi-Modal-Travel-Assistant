# Run book

Every command needed to go from a fresh machine to a running, tested application.
Each one is literally pasteable. Windows and macOS/Linux variants are given where
they differ.

**The application runs with no API keys.** Steps 3a and onwards work on a machine
that has never seen an API key in its life. Adding keys is optional and covered in
step 3b.

---

## 1. Get the code

**Windows (Command Prompt or PowerShell)**

```
cd %USERPROFILE%\Desktop
git clone <your-repository-url> Multi-Modal-Travel-Assistant
cd Multi-Modal-Travel-Assistant
```

**macOS / Linux**

```bash
cd ~
git clone <your-repository-url> Multi-Modal-Travel-Assistant
cd Multi-Modal-Travel-Assistant
```

> Substitute `<your-repository-url>` with the repository's clone URL. If you
> already have the folder, just `cd` into it.

---

## 2. Create the environment

Python 3.11 is required. The conda route is preferred because it pins the
interpreter as well as the packages.

**Windows (Command Prompt / PowerShell / Anaconda Prompt)**

```
conda env create -f environment.yml
conda activate travel-agent
```

**macOS / Linux**

```bash
conda env create -f environment.yml
conda activate travel-agent
```

**Without conda** - any platform, using a plain virtual environment:

```
python -m venv .venv
```

then activate it:

```
.venv\Scripts\activate            REM Windows Command Prompt
.venv\Scripts\Activate.ps1        # Windows PowerShell
source .venv/bin/activate         # macOS / Linux
```

and install:

```
pip install -r requirements.txt
```

**Verify the environment before going further:**

```
python scripts/smoke_test.py
```

Expected: every dependency listed with `[ OK ]`, then three LangGraph runtime
checks, then `SMOKE TEST PASSED - environment is ready.`

---

## 3a. Configure - the no-API-keys path (recommended first run)

**Windows**

```
copy .env.example .env
```

**macOS / Linux**

```bash
cp .env.example .env
```

**Then change nothing.** Every key in `.env.example` is blank, which selects the
deterministic mock providers: a mock model that emits real tool-calling payloads,
mock weather with realistic seasonal data, verified Wikimedia photographs, and
mock web search. The full application, the full test suite and the entire demo
work in this state.

## 3b. Configure - with API keys (optional)

Open `.env` and fill in whichever you have. Provider selection is automatic:
`GROQ_API_KEY` wins, then `ANTHROPIC_API_KEY`, then `OPENAI_API_KEY`, then the
mock. `LLM_PROVIDER` overrides that order entirely.

```
GROQ_API_KEY=gsk_...                 # https://console.groq.com/keys
ANTHROPIC_API_KEY=sk-ant-...         # https://console.anthropic.com/settings/keys
OPENAI_API_KEY=sk-...                # https://platform.openai.com/api-keys
```

The tools stay on mocks unless you switch them individually:

```
WEATHER_PROVIDER=live    OPENWEATHER_API_KEY=...
IMAGE_PROVIDER=live      UNSPLASH_ACCESS_KEY=...
SEARCH_PROVIDER=live     # DuckDuckGo needs no key
```

One setting is worth checking even on the no-keys path:

```
ROUTER_SIMILARITY_THRESHOLD=0.07
```

If that is not `0.07`, routing will misbehave - see the troubleshooting table.

---

## 4. Build the vector store

```
python scripts/seed_vectorstore.py
```

Expected: 9 chunks each for New York, Paris and Tokyo (27 total), then a
similarity matrix. Read the bottom of it - seeded cities should score 0.10 to
0.21, unseeded ones 0.00 to 0.04, and the suggested threshold should be near
0.07.

The command is idempotent; running it again prints `Store is already up to date`.
Force a rebuild with `--force`, or use the pure-Python backend with
`--backend numpy` if FAISS gives trouble.

---

## 5. Run the tests

```
pytest -q
```

Expected: `327 passed`. No API keys are needed, and no network access is
required.

Useful variants:

```
pytest -q tests/test_graph.py            REM the graph and the parallel fan-out
pytest -q tests/test_tool_executor.py    REM the manual tool executor
pytest -q tests/test_memory.py           REM the checkpointer and follow-up skip
pytest -q --cov=travel_agent             REM with coverage
```

---

## 6. Generate the graph diagram

`graph.png` and `graph.mmd` are already committed, so this is only needed if you
change the topology:

```
python scripts/export_graph.py
```

Expected: `graph.png and graph.mmd already exist ... Nothing to do.`

To regenerate them (needs network access, as the renderer is a web service):

```
python scripts/export_graph.py --force
```

If the render fails, `graph.mmd` is still written and can be pasted into
<https://mermaid.live>.

---

## 7. Launch the application

```
streamlit run src/travel_agent/ui/app.py
```

Then open:

**<http://localhost:8501>**

Streamlit usually opens it for you. If port 8501 is busy, add
`--server.port 8502` and open that instead.

---

## 8. The demo, in order

Four queries. Each takes about one second on mocks and exercises a different path.

### Query 1 - a city in the knowledge base

Type: **`Tell me about Tokyo`**

You should see:
- A summary, four photographs, and a seven-day forecast chart with a shaded
  high/low band.
- In the trace panel under **Routing**: *Source: Internal knowledge base*, decided
  by *exact name match on 'Tokyo'*, similarity **0.207** against a threshold of
  **0.07**, and a table showing Tokyo above the threshold with New York and Paris
  below it.
- Under **Parallelism**: sequential-equivalent around 1900-2000 ms against an
  actual wall clock around 1000-1300 ms, roughly **1.7x**.

### Query 2 - the follow-up

Type: **`what about next week?`**

You should see:
- The heading still says **Tokyo** - the city came from memory, not from the text.
- The forecast dates have moved forward by a week.
- The photographs are unchanged, because they were not re-fetched.
- Under **Memory**: *Skipped: retrieve_vector, execute_images*, with the work
  avoided in milliseconds and a note that this is a saving in work and API cost
  rather than wall clock.

### Query 3 - a city that is not in the knowledge base

Type: **`Tell me about Kyoto`**

You should see:
- *Answered from a live web search* under the heading.
- Under **Routing**: Kyoto scores about **0.040**, below the 0.07 threshold, so
  the graph switched to web search.
- Source links beneath the summary.
- Placeholder gallery imagery, captioned honestly as representative rather than
  photographs of Kyoto.

### Query 4 - break the weather API

In the sidebar, turn on **Break the weather API**, leave the mode as
`server_error`, then type: **`Tell me about Paris`**

You should see:
- The summary and the gallery render completely.
- A yellow banner: *get_weather_forecast unavailable: simulated HTTP 500 from
  provider*.
- The forecast area replaced with an explanation, not a stack trace or an empty
  panel.
- Under **Tools**: the weather tool marked `error`, the others `ok`.

Turn the toggle off afterwards.

### Optional - the memory boundary

Change **Thread id** in the sidebar to `another-conversation`, then type
**`what about next week?`**. With no history on that thread, the assistant asks
which city you mean rather than guessing.

---

## 9. Verifying a live provider (optional)

With a key configured, one deliberate request end to end:

```
python scripts/live_smoke_test.py
```

Expected: the model availability check, the route and score, the tools that
executed, a validated `TravelResponse`, the token usage, and the summary the live
model wrote.

---

## 10. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'travel_agent'` | The environment is not active, or the shell was opened before it was created. | `conda activate travel-agent`. Confirm with `python -c "import travel_agent"`. |
| `vector store not found ... Run: python scripts/seed_vectorstore.py` | Step 4 was skipped. The app shows this as a red banner and routes everything to web search. | `python scripts/seed_vectorstore.py` |
| Every city routes to web search, even Tokyo | `ROUTER_SIMILARITY_THRESHOLD` in `.env` is too high. No city can score 0.55; the calibrated value is 0.07. | Set `ROUTER_SIMILARITY_THRESHOLD=0.07`. The app also shows a warning banner naming the measured band. |
| `ImportError` or a DLL error mentioning `faiss` | The FAISS native wheel does not load on this machine. | `python scripts/seed_vectorstore.py --force --backend numpy`, then set `VECTOR_STORE_BACKEND=numpy` in `.env`. Behaviour is identical. |
| The gallery shows plain gradient images saying "offline fallback" | Wikimedia Commons is unreachable, so the bundled images are being used. | Nothing is broken. Force remote with `IMAGE_FALLBACK_MODE=remote`, or leave it - this is the intended degradation. |
| `The request failed: APIConnectionError` | A live provider is unreachable or rate limiting. | Retry; the app stays usable. To remove the dependency entirely, set `LLM_PROVIDER=mock`. |
| The app hangs after a question and never returns | Almost always a live provider not responding. The turn has a 120-second timeout, after which the UI reports it and stays usable. | Wait for the timeout, then set `LLM_PROVIDER=mock` to confirm the graph itself is healthy. |
| `Port 8501 is already in use` | Another Streamlit instance is running. | `streamlit run src/travel_agent/ui/app.py --server.port 8502` |
| Tests fail with a timeout on a slow machine | The parallel-timing tests assert wall clock against branch durations. | Re-run; they carry wide margins. If it persists, run `pytest -q -k "not parallel"` to confirm everything else passes. |
| Conversations disappear when the app restarts | `MemorySaver` is the default and lives in the process. | Set `CHECKPOINTER=sqlite` in `.env` for durable memory in `data/checkpoints.sqlite`. |
| Console output shows `?` instead of punctuation | Windows console encoding (cp1252) cannot render some characters a live model emits. | Cosmetic only. `set PYTHONUTF8=1` before running if it bothers you. |

---

## Command summary

```
conda env create -f environment.yml
conda activate travel-agent
python scripts/smoke_test.py
copy .env.example .env                     REM cp on macOS/Linux
python scripts/seed_vectorstore.py
pytest -q
streamlit run src/travel_agent/ui/app.py
```

Then type, in order: `Tell me about Tokyo` -> `what about next week?` ->
`Tell me about Kyoto` -> toggle **Break the weather API** -> `Tell me about Paris`.
