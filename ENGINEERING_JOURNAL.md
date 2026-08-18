# Engineering Journal

A working record of how this project was built, written so I can explain any part
of it out loud without re-reading the code.

> **Status:** build steps 1-6 complete (environment, foundations, schemas,
> services, seed corpus, vector store). The tool layer, graph, UI and docs are
> still to come. Sections marked *(pending)* will be filled in as those land.

---

## 1. What this project is, in 30 seconds

It is a travel assistant that answers a question like "Tell me about Kyoto" with
a written summary, a photo gallery and a weather chart. The interesting part is
not the answer, it is the machine that produces it: a **LangGraph** state machine
that decides for itself where to get the facts. If the city is one it has
documents about, it reads its own knowledge base; if not, it searches the web.
While that is happening it fetches weather and images **at the same time**, not
one after the other. It remembers the conversation, so a follow-up like "what
about next week?" re-runs only the weather step instead of redoing everything.
Every external service sits behind an interface with a real implementation and a
mock one, so the whole thing runs with no API keys at all.

### Vocabulary, one line each

Terms used throughout this document, defined the first time they matter.

| Term | What it means here |
|---|---|
| **Graph** | The whole program-as-a-diagram: a set of steps and the rules for moving between them. |
| **Node** | One step. A Python function that takes the current state and returns an update to it. |
| **Edge** | An arrow from one node to the next. |
| **Conditional edge** | An arrow whose destination is decided at runtime by a function. This is how routing works. |
| **State** | One typed dictionary carrying everything known about the request, passed from node to node. |
| **Reducer** | A function that says how to merge two writes to the same state key. Required when nodes run in parallel. |
| **Superstep** | One "tick" of the graph. Every node scheduled in the same superstep runs concurrently. |
| **Checkpointer** | Storage that saves the state after each step, so a later turn can continue from it. |
| **Thread id** | The key a checkpointer files a conversation under. |
| **Embedding** | A list of numbers representing text, arranged so similar texts sit close together. |
| **Vector store** | A place to keep embeddings and search them by similarity. |
| **Cosine similarity** | The closeness score between two embeddings. 1.0 = same direction, 0.0 = unrelated. |
| **Chunk** | One passage of text, the unit that gets embedded and retrieved. |
| **Tool call** | A structured request from the model - a function name plus JSON arguments - that the application executes. |
| **TF-IDF** | Term frequency x inverse document frequency: weight words by how often they appear here and how rare they are overall. |

---

## 2. Step 1 - Environment, and checking the API before trusting it

**What I did.** Created a conda environment called `travel-agent` on Python 3.11,
installed pinned versions of everything, then wrote `scripts/smoke_test.py` to
import every dependency and print its version.

**Why 3.11 and not the system Python.** The machine's default is 3.13. Native
wheels (FAISS in particular) are reliably available for 3.11 and sometimes lag on
the newest release. Pinning the interpreter removes a whole class of "works on my
machine" failure.

**Why I did an API recon before writing any node code.** The plan's biggest
identified risk was that LangGraph's API has moved since the version I had in my
head. Instead of writing code and finding out, I introspected the *installed*
packages with `inspect.signature` and confirmed four things that the architecture
depends on:

| Question | Verified answer |
|---|---|
| Can a conditional edge return a **list** of node names, to fan out? | Yes: the signature is `Callable[..., Hashable \| Sequence[Hashable]]`. |
| Do parallel nodes actually run concurrently? | Yes: three nodes sleeping 1.0s each finished in **1.02s** wall clock. |
| Are reducers genuinely required for concurrent writes? | Yes: without one, LangGraph raises `InvalidUpdateError: Can receive only one value per step`. |
| What does `ToolMessage` look like? | It has `content`, `tool_call_id`, `name` **and `status`** - the last one matters (section 6). |

Those four facts are now permanently asserted in `scripts/smoke_test.py`, so if a
future dependency bump breaks one of them, the smoke test says so immediately
rather than the app failing strangely.

**What I chose it over.** I could have read the documentation. Documentation
describes the latest release; the installed version is the one that runs. For a
one-day build with a hard deadline, ten minutes of introspection is cheaper than
one hour of debugging a signature change.

---

## 3. Step 2 - Why Groq is the default model driver

**What I did.** Implemented four LLM drivers behind one interface - Groq,
Anthropic, OpenAI and a deterministic mock - and made the selection automatic:
an explicit `LLM_PROVIDER` wins if set, otherwise the first API key present in
the order Groq, Anthropic, OpenAI, and with no keys at all it falls back to the
mock.

**Why Groq is the default, given the assignment names OpenAI and Anthropic.**

1. Both named providers are **fully implemented and covered by the same
   interface**. Switching is one line in `.env` (`LLM_PROVIDER=anthropic`). Groq
   is a development and demo convenience, not a substitute for the requirement.
2. The free-tier token allowance on OpenAI and Anthropic does not comfortably
   cover a day of iterative development *plus* a live demo in front of a panel.
   Groq's does. A demo that fails on a quota error in the room is worse than any
   architectural nicety.
3. Groq's API is OpenAI-compatible and returns a **genuine `tool_calls`
   payload**. That matters specifically for Distinction 1: the manual tool
   executor is exercised against the real wire protocol, not a shim that fakes
   it.
4. Groq's inference is fast, so the parallel fan-out measurement is dominated by
   tool latency rather than model latency. That makes the speed-up number a
   cleaner measure of what the graph does.

**Implementation route chosen: `langchain-groq` (`ChatGroq`), version 1.1.3.** I
checked its dependency constraints before committing: it requires
`langchain-core>=1.4.0,<2.0.0`, and this project pins `langchain-core==1.5.6`, so
there is no conflict. The fallback plan - drop to the raw `groq` SDK and adapt
its responses to `AIMessage`/`ToolMessage` by hand - was not needed. Staying on
`ChatGroq` means all four drivers speak native `langchain-core` message types and
the graph code never learns which provider it is talking to.

**Model id: `openai/gpt-oss-120b`.** I did not pick this from memory. Groq's
model documentation lists it on the *production* tier (not preview) with both
tool calling and JSON mode supported. I could not query `/v1/models` to confirm
availability because there was no API key on the machine at the time - the
endpoint returns 401 unauthenticated. Two mitigations:

* the id lives in settings as `GROQ_MODEL`, changeable in one place;
* at start-up, when a key *is* present, the Groq driver queries `/v1/models`,
  logs whether the configured id is still listed, and falls back to an available
  tool-capable model rather than hard-failing. The check result is pushed into
  the UI trace so it is visible, not buried in a log.

**Structured output on Groq.** I do not assume schema-constrained decoding is
supported. The path is JSON mode plus the project's own Pydantic
validate-and-repair pass, which is the safer route and is applied uniformly
across all providers rather than being special-cased.

---

## 4. Steps 3-4 - Foundations and the typed state

**What I did.** Built the unglamorous layer first: a custom exception hierarchy,
structured logging with timing helpers, centralised settings, a retry utility,
and then the Pydantic models and the graph state.

**Why settings are a typed model rather than `os.getenv` calls.** Everything
tunable is declared once in `config/settings.py` with a type and a default. A bad
value fails at start-up with a clear message instead of halfway through a
request. No other module reads the environment.

One small thing that turned out to matter: `.env.example` ships every key as
`GROQ_API_KEY=` with nothing after the equals sign, so reviewers can see the full
list of options. Pydantic reads that as the empty string, which is *truthy
enough* to look configured. A validator converts blank strings to `None`, so an
empty key correctly means "not set" and the app falls back to the mock instead of
calling a live API with no credential.

### The state, and why reducers are the interesting part

`TravelState` is a `TypedDict` with about twenty keys: the conversation, the
resolved slots (city, date range), the routing decision and its score, the
fan-out results, the observability data, and the final response.

The part worth explaining to a panel is this. When two nodes run at the same time
and both write the same key, LangGraph does **not** silently pick a winner - it
raises an error. A reducer is the function that says how to merge those writes,
attached to the key with `Annotated[...]`:

```python
trace: Annotated[list[TraceEvent], append_list]      # concatenate
timings: Annotated[dict[str, float], merge_timings]  # union of keys
token_usage: Annotated[TokenUsage, add_token_usage]  # field-wise sum
images: Annotated[list[ImageAsset] | None, replace_value]  # last writer wins
```

So the reducers are not decoration - **the parallel fan-out is only legal because
they exist**. That is why `tests/test_reducers.py` tests each one directly *and*
then runs a real three-branch LangGraph fan-out and asserts the merged result,
plus one test proving that an un-reduced key really does raise. If someone later
"simplifies" the annotations away, a test fails and explains why.

Two design details I would be asked about:

* **`None` means reset.** An accumulating reducer can only grow. Each new turn
  needs a clear trace, so passing `None` for those keys is the documented reset
  signal (`new_turn_updates` does this). Slots and previous results are
  deliberately *not* cleared - that is what makes follow-up turns work.
* **`replace_value` for single-writer keys.** Weather, images and knowledge each
  have one writer per turn, so they do not need to accumulate. They still carry a
  reducer, because on a follow-up turn only the weather branch runs and this
  keeps the previous turn's images in state instead of wiping them.

---

## 5. Step 5 - Retrieval: embeddings and the vector store

**What I did.** Wrote a dependency-free TF-IDF embedder, two vector store
backends behind one interface (FAISS with an automatic NumPy fallback), and a
retrieval service that ties them together.

**Why not `sentence-transformers`.** It pulls in PyTorch: hundreds of megabytes,
a slow cold start, and a download that can fail on a reviewer's machine. The
retrieval problem here is narrow - decide whether a query names one of three
seeded cities - and that is a lexical question, which TF-IDF answers well. The
trade-off is honest and I will say it out loud: this embedder does not know that
"the French capital" means Paris. If that mattered, `EMBEDDING_PROVIDER=openai`
swaps in real semantic embeddings without touching the store or the router.

**Why FAISS with a NumPy fallback.** FAISS is the library the assignment names,
it is one wheel, and `IndexFlatIP` is an *exact* index - no approximation to
reason about. But a native wheel is exactly the sort of thing that fails to
install on someone else's laptop, so `NumpyVectorStore` implements the identical
interface with a brute-force dot product. If the FAISS import fails, the factory
logs it and returns the NumPy store; a test forces that path. At 27 chunks brute
force is not measurably slower, so the honest justification for FAISS is that it
is the right shape for a corpus that grows, not that it is faster today.

**Why inner product rather than L2 distance.** Every vector is normalised to unit
length before it is stored, and for unit vectors the inner product *is* the
cosine similarity. That means the number FAISS returns is directly comparable
with the router threshold, with no conversion step to get wrong.

**Chunking.** The corpus is markdown with one `##` section per topic, so one
section becomes one chunk: roughly 100-150 words, self-contained, about a single
subject. The section heading is kept as metadata *and* prepended to the embedded
text, because "Getting around" is a strong retrieval signal for a transit
question and throwing it away would be free information lost.

---

## 6. Problems I hit and how I solved them

The honest list, in the order they happened. None of these are invented.

### 6.1 A sync node that returns a coroutine is not an async node

**Symptom.** The first version of the smoke test registered its parallel nodes as
lambdas that called an `async def` helper:

```python
builder.add_node("weather", lambda state: _sleeper("weather"))
```

It failed with:

```
InvalidUpdateError: Expected dict, got <coroutine object _sleeper at 0x...>
sys:1: RuntimeWarning: coroutine '_sleeper' was never awaited
```

**Why LangGraph behaved that way.** LangGraph inspects the callable you register
to decide whether to `await` it. A `lambda` is an ordinary synchronous function -
the fact that its *return value* happens to be a coroutine is invisible to that
check. So LangGraph called it, got a coroutine object back, and tried to merge
that object into the state as if it were a state update. The "never awaited"
warning is the same bug seen from Python's side: nothing ever ran the coroutine.

**Fix.** Register coroutine *functions* directly:

```python
async def _weather(state: FanOutState) -> dict[str, list[str]]:
    await asyncio.sleep(0.5)
    return {"trace": ["weather"]}

builder.add_node("weather", _weather)
```

**Why it matters beyond the smoke test.** Every fan-out node in this project is
`async def`. If one were accidentally wrapped in a sync lambda, it would not run
concurrently - it would fail outright, which is the good outcome. The failure is
loud rather than silent, which is worth knowing.

### 6.2 Windows console encoding mangled the output

**Symptom.** The smoke test printed `SMOKE TEST PASSED ? environment is ready.`
The em-dash came out as a replacement character.

**Cause.** The Windows console defaults to the cp1252 code page, which has no
mapping for U+2014. Python encodes to the console's code page on write.

**Fix.** Replaced non-ASCII punctuation in console output with ASCII equivalents,
and checked the file for any remaining non-ASCII characters. I could have forced
UTF-8 with `PYTHONUTF8=1` or reconfigured `sys.stdout`, but that pushes an
environment requirement onto whoever runs it. Plain ASCII in terminal output has
no downside, and the reviewer is likely to run this on Windows.

### 6.3 A Pydantic field named `date` shadowed the `date` type

**Symptom.** Importing the response schemas failed at class-creation time:

```
PydanticUserError: Error when building FieldInfo from annotated attribute.
Make sure you don't have any field name clashing with a type annotation.
```

**Cause.** `ForecastPoint` declared `date: date`. Inside the class body the name
`date` is being bound as a field, so when Pydantic resolves the annotation string
`"date"` it finds the field, not `datetime.date`.

**Fix.** Import the type under a different name - `from datetime import date as
DateType` - and annotate `date: DateType`. The field keeps the name the UI and
the JSON contract want; only the annotation changes.

### 6.4 The router ranked Kyoto above Paris

This is the most instructive failure of the build, and the one I would volunteer
in an interview.

**Symptom.** A test asserting that seeded cities outscore unseeded ones failed:

```
assert 0.136 < 0.117    # "Kyoto" scored HIGHER than "Paris"
```

**Cause.** Two separate faults, uncovered one after the other.

*Fault one: scoring on the single best chunk.* The router took the highest cosine
similarity between the query and any individual chunk. My Tokyo corpus mentions
Kyoto once, in the day-trips section ("Kyoto is reachable in 2 hours 15 minutes
by shinkansen"). One strong sentence in one chunk therefore beat Paris, a city
discussed across nine chunks. The signal I was measuring was "does this word
appear anywhere in my corpus?" when the question I actually needed answered was
"is this city a *subject* of my knowledge base?".

*Fault two: out-of-vocabulary tokens produced pure noise.* With the hashing
trick, a token maps to a bucket whether or not the corpus contains it. Standard
TF-IDF gives an unseen term the *maximum* inverse-document-frequency weight
because it is maximally rare. Combined, those two rules meant an unknown word
became the loudest component of the query vector, landing on whichever bucket it
happened to hash to. Measured result: **"Reykjavik" scored 0.103 against Paris**,
higher than several genuine matches. That number was entirely hash collision.

**Fix, in two parts.**

1. **Score against a city profile, not a chunk.** Each city gets one vector: the
   normalised mean of its chunk vectors. A word used throughout a city's
   documents survives the average; a word mentioned once in nine passages is
   diluted to about a ninth of its weight. Chunk-level search is still used for
   *retrieving* passages - it is only the routing signal that changed.
2. **Drop unknown tokens at query time.** If a token was never seen while
   fitting, it cannot carry real signal, only collision noise, so it contributes
   nothing. Unknown cities now score exactly **0.000**.

**Measured result** (from `python scripts/seed_vectorstore.py`):

```
city probe               New York        Paris        Tokyo     best
Paris                       0.000        0.102       -0.004    0.102  (in store)
Tokyo                      -0.049       -0.122        0.207    0.207  (in store)
New York                    0.164        0.088       -0.045    0.164  (in store)
Kyoto                      -0.033        0.037        0.040    0.040
Snohomish                   0.000        0.000        0.000    0.000
Reykjavik                   0.000        0.000        0.000    0.000
Bogota                      0.000        0.000        0.000    0.000

  lowest score for a KNOWN city    : 0.102
  highest score for an UNKNOWN city: 0.040
  separation margin                : 0.062
  midpoint (suggested threshold)   : 0.071
```

Both faults are now pinned down by regression tests
(`test_city_centroid_beats_naive_max_chunk_scoring`,
`test_unknown_vocabulary_scores_exactly_zero`), each carrying a comment
explaining the bug it prevents.

### 6.5 The routing threshold was a guess, and the guess was wrong

**Symptom.** My plan proposed `ROUTER_SIMILARITY_THRESHOLD = 0.55`. Once the
scoring was correct, the highest score any seeded city achieved was **0.207**. A
0.55 threshold would have sent *every* city to web search, and the vector-store
path - a core requirement - would never have executed.

**Cause.** I picked 0.55 because it "sounds like a confident match", which is
what people do with thresholds when they have not looked at the data. Cosine
similarity between a one-word query and a whole-city profile is inherently small:
the query is a single direction, the profile is an average over nine passages
covering nine topics, so even a perfect topical match only recovers a fraction of
the profile's magnitude.

**Fix.** Derive it from measurements. The seeder prints the full matrix and the
gap between the lowest known-city score (0.102) and the highest unknown-city
score (0.040), and the default is now **0.07**, which sits inside that gap.
`evals/run_eval.py` will sweep the threshold across a labelled query set to
confirm the choice rather than assert it.

**The lesson I would state out loud:** the absolute value of a similarity score
is close to meaningless on its own. What matters is the *separation* between the
two populations you are trying to distinguish, and you only find that by
measuring.

### 6.6 The router must score the extracted city, not the raw sentence

**Symptom.** Even after the fixes above, `"what is the weather in Kyoto next
week"` scored **0.089** against Paris - dangerously close to the 0.102 achieved
by a genuine Paris query.

**Cause.** The words carrying that score were "weather", "next" and "week" -
ordinary vocabulary that appears throughout every city's documents. "Kyoto"
contributed nothing (correctly, it is out of vocabulary). So the score was
measuring how travel-shaped the sentence was, not which city it was about.

**Fix.** The router scores the **extracted city slot**, not the raw user text.
The intent node pulls the city out first; the router embeds just `"Kyoto"`, which
scores 0.040 and routes to web search correctly. This is now documented directly
on `KnowledgeRetriever.best_match`, and the seeder prints both numbers side by
side so the reasoning is visible rather than folded away.

A gazetteer sits in front of the similarity check as well: an exact or alias
match on a known city name (`"NYC"`, `"New York City"` to `"New York"`) resolves
immediately. Whether the knowledge base has documents for a city is a question
about *names*, and answering a naming question with a cosine threshold is how you
end up explaining to a panel why "NYC" went to web search.

### 6.7 `graph.png` swapped a Graphviz dependency for a network dependency

**Symptom (anticipated, then confirmed).** There is no Graphviz on this machine
(`dot: command not found`). LangGraph's `draw_mermaid_png()` defaults to
`MermaidDrawMethod.API`, which renders by calling the mermaid.ink web service.

**Why that is not a fix.** I initially recorded this risk as mitigated. It is
not: it replaces "needs a local binary" with "needs the internet at the moment
the reviewer runs it". `graph.png` is a **required submission artifact**, so it
has to exist regardless of network conditions.

**Fix.** Generate `graph.png` and `graph.mmd` once and **commit them to the repo
root**, so they are present in a clone whatever happens.
`scripts/export_graph.py` will skip regeneration when the files already exist
unless `--force` is passed, and keeps a local/ASCII fallback for the no-network
case. *(Pending: lands with the graph build step.)*

### 6.8 Heredocs kept failing on long Python files

**Symptom.** Writing large modules through shell heredocs intermittently died
with `unexpected EOF while looking for matching quote`, mid-file.

**Fix.** Switched to writing files directly rather than piping them through the
shell. Not an application bug, but it cost real time, so it is recorded here.

---

## 7. Requirement traceability *(in progress)*

| Assignment requirement | Where it lives | One-line explanation |
|---|---|---|
| Vector store seeded with 3 cities | `data/city_facts/*.md`, `scripts/seed_vectorstore.py` | 27 chunks across Paris, Tokyo and New York, 9 each. |
| Typed state | `src/travel_agent/schemas/state.py` | `TypedDict` with `Annotated` reducers on every concurrently-written key. |
| Structured output object | `src/travel_agent/schemas/response.py` | `TravelResponse` with `city_summary`, `weather_forecast`, `image_urls`. |
| Manual tool execution (Distinction 1) | `graph/nodes/tool_executor.py` | *(pending)* |
| Parallel fan-out (Distinction 2) | `graph/edges.py`, `graph/builder.py` | *(pending)* |
| Checkpointer + follow-up (Distinction 3) | `graph/builder.py`, `graph/nodes/classify_intent.py` | *(pending)* |
| Conditional edge on knowledge availability | `graph/edges.py` | *(pending)* |
| Streamlit UI with chart and gallery | `ui/app.py` | *(pending)* |
| `graph.png` | repo root, `scripts/export_graph.py` | *(pending)* |

---

## 8. Things I deliberately did not do *(running list)*

* **No `sentence-transformers` by default** - the download and cold start are not
  worth it for a three-city corpus. The interface supports it.
* **No `langchain-community`** - the LangChain 1.x package split makes it a
  version-drift risk, and nothing here needs it.
* **No `tenacity`** - a 30-line retry utility I can explain line by line is worth
  more in an interview than a dependency, and it let me implement
  `Retry-After` handling exactly the way Groq's rate limiting needs.
* **No dollar-cost display** - Groq, OpenAI and Anthropic price differently. The
  UI shows token counts and names the provider rather than printing a number that
  would be wrong for two of the three.

---

## 9. Interview Q&A *(pending - written once the graph and UI land)*
