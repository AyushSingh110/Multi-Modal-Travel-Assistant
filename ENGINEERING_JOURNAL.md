# Engineering Journal

A working record of how this project was built, written so I can explain any part
of it out loud without re-reading the code.

> **Status:** build steps 1-10 complete (environment, foundations, schemas,
> services, seed corpus, vector store, layered router, tool layer, manual tool
> executor, graph assembly, parallel fan-out). The checkpointer, the UI and the
> docs are still to come. Sections marked *(pending)* fill in as those land.

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

**The lesson I would state out loud:** the absolute cosine value means nothing;
only the separation between the two populations you are trying to distinguish
means anything, and you only get that by measuring.

Two follow-on consequences, both now built:

* the router no longer relies on that single number alone (section 7.1);
* a start-up guard measures the separation on every run and makes this failure
  impossible to reproduce silently (section 7.2).

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

## 7. Hardening the router

### 7.1 Why layered, not a single threshold

The measured separation is real but thin: seeded cities score 0.10 to 0.21,
unseeded ones 0.00 to 0.04. A single gate at 0.07 works on the probes I measured,
but it is fragile in two specific ways. Anything landing in the 0.04-0.10 band is
decided by a hair. And "NYC" - a city the knowledge base covers in nine
documents - would fail it outright, because "nyc" is not a token in the corpus at
all.

So the router now asks a cheaper, more certain question first:

| Layer | Question | Outcome | match_reason |
|---|---|---|---|
| 1. Gazetteer | Does this name, or an alias of it, match a city I hold documents for? | vector store | `exact` |
| 2. Similarity | Is the centroid cosine above the threshold? | vector store or web | `similarity` |
| 3. Nothing | Was any city resolved at all? | clarify | `none` |

The reasoning: whether the knowledge base has documents about a city is, in the
ordinary case, **a question about names**, and a name lookup answers it exactly.
Answering a naming question with a cosine threshold is how you end up explaining
to a panel why "NYC" went to web search. Only when the name is unrecognised does
the similarity score have to make a judgement call - and that is the case it is
genuinely good at, because an unseen city name scores 0.000.

Name folding is Unicode-aware rather than a lookup table of accented spellings.
NFKD decomposition splits an accented character into a base letter plus a
combining mark, and the marks are discarded, so "Zurich", "zürich" and "ZÜRICH"
fold to one key - and it works for names nobody remembered to add.

**Crucially, layering does not hide the measurement.** An exact match still
records the similarity score, the threshold, and every city's score, so the UI
can say:

> routed to vector store (exact match on 'Tokyo'; similarity 0.207, threshold 0.07)

The score became *displayed confidence* rather than a single fragile gate.

**The trade-off, stated plainly.** The gazetteer needs maintenance: every new city
and nickname added to the corpus is another entry, and one that drifts out of date
stops catching things without complaining. The similarity path is what
generalises, and it still decides every case the gazetteer has not been taught. At
this corpus size layering is clearly right; past a few hundred cities the
gazetteer should be *generated* from the corpus instead of hand-maintained.

### 7.2 The start-up guard, or making a silent failure loud

The 0.55 threshold bug had one property that made it dangerous: it failed
**silently**. Every city would route to web search, every answer would still
render, and nothing would say the vector store had been skipped. The app looked
healthy.

`KnowledgeRouter.check_threshold` now measures the separation at start-up - the
lowest score among cities the store covers, the highest among control cities it
does not - and classifies the configured threshold:

* **too_high**: at or above the lowest seeded-city score. The message names the
  city, the measured band, the value that would be correct, and the exact setting
  to change.
* **too_low**: at or below the highest unseeded-city score, so cities the store
  knows nothing about could be answered from it.
* **ok**: inside the band, with the margin reported.

Anything other than `ok` is logged as a fenced warning block, and the diagnostics
are a Pydantic model so the UI can display the same verdict rather than leaving it
in a terminal nobody is watching. A test asserts the exact 0.55 case is caught.

---

## 8. Step 7 - The tool layer

### 8.1 Mock data that survives being looked at

The mocks are the demo. If the forecast chart looks fake, the architecture behind
it does not get the benefit of the doubt, so both mock providers are built to be
inspected.

**Weather.** Not `random.uniform(10, 25)`. Each city carries a real monthly
climate table, and three properties of actual weather are reproduced: a seasonal
baseline interpolated between months, day-to-day persistence through smooth sine
walks rather than independent daily draws, and conditions derived from the
precipitation number so the labels and the chart cannot disagree. Output is
deterministic per city and start date, which matters when a panel asks you to run
it again. Measured:

```
Tokyo from 2026-08-19          Tokyo from 2026-01-19
  24.0 / 31.2 C   41%            3.3 / 11.1 C   10%
  23.4 / 30.3 C   43%            3.0 / 10.5 C    9%
  22.8 / 29.5 C   45%            2.5 /  9.8 C    9%
```

Tests assert the series is neither flat (standard deviation above 0.5) nor
implausibly jumpy (no day-to-day swing above 8 C), that August in Tokyo is more
than 12 C warmer than January, and that New York in January is colder than Tokyo
in January.

**Images.** Every curated URL was verified with a real HTTP request returning
`200 image/jpeg`, because a gallery of broken thumbnails is worse than no gallery
at all. Cities with no curated set get seeded placeholder photography whose
caption says exactly that - captioning a stock photo as Kyoto would be a lie the
UI then repeats to the user.

### 8.2 Four failure modes, not one flag

The rubric asks whether the app survives a failing weather API. The honest answer
is a toggle that breaks it live in front of the reviewer, so failure injection is
built into the mocks rather than bolted on afterwards - and it simulates four
shapes, because they fail differently:

| Mode | Simulates | Retried? |
|---|---|---|
| `timeout` | provider hangs past the deadline | yes |
| `server_error` | HTTP 500 | yes |
| `rate_limit` | HTTP 429 with `Retry-After` | yes, honouring the header |
| `malformed` | HTTP 200 with an unusable body | **no** |

The last row is the one worth talking about.

**Why not just retry everything?** Because retrying only helps when the failure
is *transient* - when the same call, made again, might succeed. A timeout, a 500
and a 429 are all transient: the provider is overloaded, or the network hiccupped,
and a second attempt is a genuinely different roll of the dice.

A malformed payload is not. The provider answered successfully; it returned HTTP
200 and a body we cannot parse. Nothing about the request was wrong, so making it
again produces the identical unusable response. Retrying it three times spends
three times the user's waiting time and three times the provider's quota to
arrive at exactly the same failure - and on a rate-limited free tier, those wasted
calls can push the *next* legitimate request over the limit.

The same reasoning governs argument validation errors: if the model sent
`days=99`, sending it again unchanged fails again. That failure needs to go back
to the *model* so it can correct itself, not back to the provider.

So the retry decision is encoded in the exception hierarchy rather than in a list
of status codes at the call site: `RetryableError` and its subclasses are retried,
everything else is raised immediately. `MalformedPayloadError` deliberately does
not inherit from `RetryableError`, and a test asserts that zero retries are
attempted for it.

### 8.3 Where degradation actually lives

`ToolRegistry.execute` **never raises**. It returns a `ToolResult` carrying either
a payload or an error, alongside the duration, attempt count and provider name.
That is a deliberate boundary: deciding *the graph survives this* belongs in one
place, while turning an error into a `ToolMessage` the model can read is the
executor node's job in the next step. A test proves the rubric case directly -
with weather failing, images and search still return their data.

### 8.4 Two things that surprised me here

**The gallery was going to be 25 MB.** The verified Commons photographs are
originals: the Eiffel Tower image alone is 5.3 MB, so one city's gallery meant
roughly 25 MB of downloads. Commons accepts a `width` parameter that returns a
scaled rendition, and the same photo drops to 339 KB. I only caught it because I
checked the response *size*, not just the status code.

**A shell loop silently broke my URL verification.** Re-checking all sixteen URLs
in a `while read` loop reported `000` - no connection - for fifteen of them,
contradicting an earlier run that returned `200`. The cause was not the URLs:
`curl` inside a `while read` loop consumes the loop's own stdin. Redirecting
`</dev/null` restored the correct result. Worth recording because the symptom
looked exactly like "the network is down" and would have sent me rewriting code
that was already correct.

---

### 8.5 The gallery must survive the demo network

**The risk.** The gallery loads photographs from Wikimedia Commons. That makes the
single most visual part of the app depend on Commons being reachable *at the exact
moment of the demo*. Blocked network, captive-portal wifi, a corporate proxy - any
of those turns the screenshot that matters into a grid of broken-image icons.

**The fix, in two parts.**

1. **Bundled fallbacks.** `data/images/` holds sixteen generated placeholder PNGs,
   about 12 KB each and 202 KB in total, committed to the repository. Every
   curated image names one, and cities with no curated set get generic ones. They
   are deliberately *not* copies of the Commons photographs: redistributing
   someone else's photograph would mean shipping their licence obligations with
   the repo. They are gradient-and-skyline placeholders that say "offline fallback
   image" on their face, and they exist to keep the layout intact rather than to
   pretend to be photography.
2. **A one-shot reachability probe.** `IMAGE_FALLBACK_MODE=auto` (the default)
   probes Commons once per process with a 2.5 second timeout and caches the
   answer; `remote` and `local` force either side for testing and for a guaranteed
   offline demo. Every `ImageAsset` carries both a URL and a local path, and
   `display_source` picks between them - falling back only when the probe failed
   *and* a bundled file actually exists, so a missing fallback can never replace a
   working URL.

**The bug this uncovered, which is the interesting part.** The first version of
the probe reported Commons as unreachable on a perfectly healthy network. The
cause was not connectivity: Wikimedia's user-agent policy rejects generic library
agents, so `python-httpx/0.28.1` was getting **HTTP 403**.

The lesson is the one worth saying out loud: **a fallback that silently misfires
is worse than no fallback at all, because it degrades a healthy demo while
reporting success.** Without the fallback, a network problem would have been
obvious - broken images, an obvious cause. With a misfiring fallback, the app
looks like it is working perfectly and simply shows worse content than it should,
and nothing anywhere says why. Every automatic degradation path needs to be
tested in the *healthy* case as well as the broken one, because the failure mode
of a safety net is that it catches you when you were not falling.

Sending a descriptive `User-Agent` fixed it, and the reason is written into the
code next to the header so nobody removes it later.

**Attribution.** `data/images/ATTRIBUTION.md` credits every photograph with its
photographer and licence, and the licences are varied - Public domain, CC BY 2.0,
CC BY 4.0, CC BY-SA 2.0/3.0/4.0. Those were read from the Commons API rather than
assumed, because guessing a licence in a submitted repository is worse than not
crediting at all. The credit string travels with each asset so the interface can
display it next to the image, which is what the CC BY licences actually require.
Fetching that metadata hit a 429 from the Commons API, which is a small joke at
the expense of section 8.2.

---

## 9. Step 8 - Distinction 1: the manual tool executor

This is the piece the assignment weights most heavily, so it gets its own
section.

### 9.1 What the raw protocol actually is

A language model cannot run code. It only produces text. So "the model used a
tool" is really a three-part exchange, and my program owns the middle part.

**Part 1 - I advertise the tools.** Along with the question, I send a list of
functions the model may request: a name, a description, and a JSON schema for the
arguments. Those schemas are generated from Pydantic models, so the thing I
advertise and the thing I validate against can never drift apart.

**Part 2 - the model asks.** Rather than answering in prose, it replies with a
structured request that arrives as an `AIMessage` whose `.tool_calls` looks like:

```python
{"id": "call_a1b2c3",              # the model's handle for THIS request
 "name": "get_weather_forecast",   # which advertised tool it wants
 "args": {"city": "Tokyo", "days": 7},
 "type": "tool_call"}
```

It has not executed anything. It has produced a request and stopped.

**Part 3 - I answer, and the id is the whole game.** I run the function and send
the result back as a `ToolMessage` carrying `tool_call_id` set to the *exact* id
from the request it answers.

The id matters because a model can ask for several tools at once - this project
routinely requests weather and images in the same turn - and my replies come back
as separate messages in whatever order the tools happened to finish. Since they
run concurrently, that is frequently *not* the order they were requested in. The
id is the only thing tying an answer to its question; position tells you nothing.

**What breaks if you get it wrong**, which is the part I would be asked:

* **Wrong id** - the model attributes the weather data to the image request, then
  confidently describes photographs of a seven-day forecast. Nothing raises. You
  just get nonsense, and it looks like a model quality problem.
* **A missing ToolMessage** - most providers, Groq and OpenAI included, reject the
  *next* request outright, because the conversation now contains a question with
  no answer. A hard API error, mid-conversation.
* **Reporting failure as success** - the subtle one. See below.

### 9.2 status="error" is not cosmetic

If a tool fails and I return the string `"error: connection timed out"` as an
ordinary tool result, the model has no way to know anything went wrong. It reads
that text as the legitimate output of the weather tool and summarises accordingly:
"the weather in Paris is error: connection timed out."

`ToolMessage` has a `status` field for exactly this. Setting `status="error"`
tells the model the call did not succeed, so it can work around the gap honestly
instead of treating the error text as data. I only found this by introspecting the
installed `ToolMessage` in the Phase 1 recon rather than assuming its shape, and
it is used for real in every failure path here.

The error *content* matters too. For an unknown tool name the message includes the
list of tools that do exist, so the model can correct itself on the next turn
instead of guessing again. For a validation failure it carries the actual Pydantic
detail - which field, what was wrong - rather than a generic "invalid arguments",
for the same reason.

### 9.3 What ToolNode would have done, and what I gained by not using it

`langgraph.prebuilt.ToolNode` does all of the above in one line. It is the right
choice in most production code. Writing it out bought three specific things:

1. **Per-tool error isolation.** Each call is executed independently through
   `asyncio.gather(..., return_exceptions=True)`, so one dead tool becomes one
   error message while its siblings return their data normally. A test asserts
   that with weather broken, the image and search calls still succeed and still
   file their payloads into state.
2. **Selective execution.** An executor instance can be told it handles only
   *some* tool names. That is precisely what lets the weather branch and the image
   branch of the parallel fan-out run the same class concurrently in different
   graph nodes, each picking its own calls out of the same `AIMessage`.
3. **Observability.** Every call emits a trace event with the tool, the id, the
   arguments, the provider, the attempt count and the duration. That is the data
   the UI trace panel renders, and a prebuilt node would not surface it.

**What I gave up**, stated plainly: I now own schema validation and id
correctness. A subtle bug in either would be silent - the wrong-id failure above
raises nothing at all. That is a real cost, and the mitigation is that both are
pinned down by tests: id pairing is verified with the weather tool deliberately
made slower than the image tool, so the results *do* arrive out of order and the
assertions would fail if pairing were positional.

### 9.4 The promise is mechanised, not remembered

A comment saying "we do not use ToolNode" is worth nothing six weeks later.
`test_no_prebuilt_tool_calling_helpers_anywhere_in_the_source` walks every Python
file in `src/` and `scripts/` and fails if `ToolNode`, `create_tool_calling_agent`
or `create_react_agent` appear, or if anything imports from `langgraph.prebuilt`.

Two details make it trustworthy rather than decorative:

* It checks **imports via the AST and identifiers via the tokeniser**, so an
  aliased or function-local import cannot slip past a plain text search.
* It **ignores comments and strings**. The executor's own docstring discusses
  ToolNode at length, and documenting a decision must not break the check that
  enforces it. A companion test asserts a deliberate violation *would* be caught,
  because a guard that cannot fail is not a guard.

### 9.5 The test I deleted instead of the code I nearly wrote

This is the story I would most want to be asked about.

Providers sometimes return tool arguments as a JSON *string* rather than a JSON
object. I knew that, so I wrote a test asserting the executor handles it: build an
`AIMessage` whose `tool_calls[0]["args"]` is the string `'{"city": "Paris"}'`, run
the node, expect a successful reply.

The test failed - but not in my code. It failed while *constructing the input*.
`langchain-core` validates `tool_calls[].args` as a dictionary when the
`AIMessage` is built, so a string argument is rejected before my node is ever
reached. The case I was defending against cannot occur at that layer.

I had two options. Add a string-handling branch to the executor so the test
passes, or accept what the failure was telling me: I had put the defence in the
wrong place. The conversion genuinely does belong somewhere - raw provider
payloads really do carry strings - but that somewhere is the provider adapter,
*below* the point where an `AIMessage` exists. The tool specs already handle it
there, and a test covers it there.

So I deleted the premise rather than writing the code. What replaced it is a test
that documents the boundary: one assertion that `AIMessage` rejects string args,
one that the specs accept them a layer down, and a docstring saying why no
defensive branch exists in between - so the next person does not "fix" the
apparent gap.

**Dead defensive code invites "what does this protect against?", and "nothing" is
a bad answer.** It also costs real money over time: every unreachable branch is
code that must be read, maintained and tested, and it quietly teaches whoever
reads it next that the layer below cannot be trusted, which spreads.

---

## 10. Steps 9 and 10 - Distinction 2: the parallel fan-out

### 10.1 What a superstep is, in plain English

LangGraph does not walk the graph one node at a time. It executes in **rounds**,
called supersteps. Everything scheduled into the same round starts together, runs
concurrently, and the next round does not begin until every node in the current
one has finished.

That last part is what makes a join node possible without writing any
synchronisation code: `join` is simply the node after the round, so it cannot run
until all three branches are done. No locks, no waiting, no counting how many
branches have reported in.

### 10.2 The mechanism: a conditional edge that returns a list

An ordinary edge always leads to the same node. A **conditional edge** runs a
function at execution time and goes wherever it names. The important detail,
confirmed in the Phase 1 recon rather than assumed, is that the function may
return a *list*:

```python
def route_and_fan_out(state) -> list[str]:
    knowledge_branch = "web_search" if state["route"] == "web" else "retrieve_vector"
    return [knowledge_branch, "execute_weather", "execute_images"]
```

Three names, one superstep, three concurrent nodes. The same function also makes
the knowledge-routing choice, so a single edge expresses both requirements the
assignment lists: the conditional route *and* the fan-out.

### 10.3 Why not asyncio.gather inside one node

This is the design question I expect to be asked, because `asyncio.gather` inside
a single node would run the same work concurrently with less machinery.

The reason is that **it would make the graph lie**. A gather-based version has one
node where there should be three. `graph.png` - a required submission artifact -
would show a straight line through a single "fetch everything" box, and a reviewer
looking at the picture would have no way to see that anything runs in parallel.
The parallelism would exist but be invisible, provable only by reading the node's
body.

There are three further practical consequences:

* **Failure isolation becomes mine to write.** With separate nodes, one branch
  failing is contained by the graph. Inside a gather I would be hand-rolling
  `return_exceptions` handling and partial-state merging.
* **The routing decision loses its home.** The knowledge branch is an XOR choice.
  Expressed as edges it is a property of the topology; expressed inside a node it
  becomes an `if` statement buried in application code.
* **Per-node timing disappears.** LangGraph attributes work to nodes. One fat node
  reports one duration, so the trace panel could not show which branch was slow.

The trade-off I accepted: more nodes, and every state key those branches touch now
*requires* a reducer. That is a real constraint, and section 10.4 is about it.

### 10.4 Why the reducers are mandatory, not stylistic

When two nodes in the same superstep write the same state key, LangGraph does not
pick a winner - it raises `InvalidUpdateError: Can receive only one value per
step`. A reducer is the function that says how to merge those writes, attached via
`Annotated[...]`.

So the reducers are not tidiness. **The fan-out is only legal because they
exist.** Two tests pin this down: one builds a fan-out over an un-reduced key and
asserts the graph refuses to run, and a control test adds the annotation to the
identical topology and asserts it succeeds. If someone later "simplifies" the
annotations away, those tests explain exactly what they broke.

### 10.5 The measured result

The claim is proved by the clock rather than asserted. `plan_tools` records a
timestamp when it dispatches the fan-out; `join` compares two numbers:

* **sequential-equivalent** - the sum of the branches' own durations, which is what
  the same work would cost run one after another;
* **parallel wall clock** - how long the superstep actually took.

Measured on the mock providers at their configured latencies (weather ~900 ms,
images ~1100 ms, search ~800 ms):

| Query | Route | Sequential-equivalent | Parallel wall clock | Speed-up |
|---|---|---|---|---|
| Tell me about Tokyo | vector store | 1964 ms | 1157 ms | **1.70x** |
| Tell me about Paris | vector store | 2056 ms | 1276 ms | **1.61x** |
| Tell me about Kyoto | web search | 2865 ms | 1035 ms | **2.77x** |

**Mean 2.03x, about 1.1 seconds saved per request.**

Two things worth noticing. The web-search query gains most, because that path has
three genuinely slow branches instead of two - the more independent work there is,
the more the fan-out buys. And the vector-store path shows `retrieve_vector` at
0 ms: reading the local index is effectively free, so on that path the fan-out is
really weather against images, and the ceiling is the slower of the two.

That is the honest way to state the result: **the speed-up is bounded by the
slowest branch**, and the numbers above are what that bound looks like in
practice. A test asserts the wall clock stays below the sum with a wide margin, so
it fails if the fan-out ever silently serialises, without being flaky on a loaded
machine.

### 10.6 Making the diagram tell the truth

LangGraph renders every conditional edge as an identical dashed arrow. Accurate,
but ambiguous: the four arrows leaving `plan_tools` include two that are
*alternatives* (vector store or web search, never both) and two that are
*concurrent* (weather and images, always together). A reviewer cannot tell which
is which, and that distinction is the entire point of this section.

So the exporter labels the edges it recognises - `concurrent`, `knowledge: in
store`, `knowledge: not in store` - and renders the labelled source. The
annotation never invents structure: it edits only edges the compiled graph
actually produced, and two tests keep it honest. One asserts the annotated diagram
has exactly the same edge set as the raw generated one; the other asserts the
committed `graph.mmd` still matches the compiled graph, so the artifact cannot
drift away from the code it documents.

### 10.7 Something that surprised me

The mock providers were built with different latencies on purpose - 900 ms for
weather, 1100 ms for images - and it turned out to matter more than expected. My
first instinct had been to give both the same delay, which would have produced a
tidy "2 seconds became 1 second, 2x". That number would have been *less*
convincing, not more: identical branch times look staged, and they hide the fact
that concurrent work is bounded by its slowest member. Different latencies produce
an untidy 1.70x, and the untidiness is what makes it look like a measurement.

---

## 11. Requirement traceability *(in progress)*

| Assignment requirement | Where it lives | One-line explanation |
|---|---|---|
| Vector store seeded with 3 cities | `data/city_facts/*.md`, `scripts/seed_vectorstore.py` | 27 chunks across Paris, Tokyo and New York, 9 each. |
| Typed state | `src/travel_agent/schemas/state.py` | `TypedDict` with `Annotated` reducers on every concurrently-written key. |
| Structured output object | `src/travel_agent/schemas/response.py` | `TravelResponse` with `city_summary`, `weather_forecast`, `image_urls`. |
| Manual tool execution (Distinction 1) | `graph/nodes/tool_executor.py` | Hand-parses `tool_calls`, validates against the Pydantic schema, dispatches, and returns `ToolMessage` with matching ids and `status="error"` on failure. No prebuilt helpers - enforced by a test. |
| Parallel fan-out (Distinction 2) | `graph/edges.py::route_and_fan_out`, `graph/nodes/core.py::join` | A conditional edge returns a list of node names, so all branches run in one superstep. Measured mean speed-up 2.03x. |
| Checkpointer + follow-up (Distinction 3) | `graph/builder.py`, `graph/nodes/classify_intent.py` | *(pending)* |
| Conditional edge on knowledge availability | `services/router.py`, `graph/edges.py` | Layered decision: gazetteer, then centroid similarity against the threshold. *(edge wiring pending)* |
| Web search path for unknown cities | `tools/search/{mock,live}.py` | Mock plus DuckDuckGo and Tavily implementations. |
| Weather tool, 5-7 day forecast | `tools/weather/{mock,live}.py` | Climate-plausible mock; OpenWeatherMap live. |
| Image retrieval | `tools/images/{mock,live}.py` | Verified Commons photographs; Unsplash live. |
| Graceful degradation when a tool fails | `tools/registry.py`, `tools/retry.py` | `execute()` returns an error result rather than raising; retries are bounded. |
| Streamlit UI with chart and gallery | `ui/app.py` | *(pending - step 13)* |
| `graph.png` | repo root, `scripts/export_graph.py` | Generated from the compiled graph, labelled to distinguish XOR from concurrent edges, and committed so it exists offline. |

---

## 12. Things I deliberately did not do *(running list)*

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

## 13. Interview Q&A *(pending - written once the UI lands)*
