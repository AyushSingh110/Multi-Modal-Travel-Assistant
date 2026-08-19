# Engineering Journal

A working record of how this project was built, written so I can explain any part
of it out loud without re-reading the code.

> **Status:** the application is complete - all three distinction challenges,
> the Streamlit interface and the model-backed synthesizer are implemented and
> tested, and the live provider path has been verified end to end. The README,
> the run book and the interview section are what remain.

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

### 6.9 An empty error message that would have looked like a rendering bug

**Symptom.** With the weather tool forced to time out, the UI warning would have
read:

    weather unavailable:

Nothing after the colon.

**Cause.** `asyncio.TimeoutError` carries no message. `str(exc)` is the empty
string, so the warning was formatted from nothing at all.

**Fix.** Every tool failure now gets a human-readable description, synthesised
from the exception type and the configured timeout when the exception itself has
nothing to say: `TimeoutError after 12s`.

**Why it matters more than it looks.** A user seeing "weather unavailable:" does
not conclude "the weather provider timed out". They conclude the app is broken -
a truncated string looks like a rendering fault, not a network condition. The
degradation path was working perfectly and still communicated the wrong thing.

**The lesson, which is the transferable part:** *assert on what the user actually
sees, not just that something happened.* The test that caught this checked the
**content** of the warning; a test asserting only `response.warnings` was
non-empty would have passed happily and shipped the bug.

### 6.10 A patch that silently did not apply

**Symptom.** I added edge labels to the graph diagram, regenerated it, and the
output had no labels.

**Cause.** I had edited the exporter with a script that did a string replacement
and printed "patched" unconditionally. The anchor text had shifted - ruff had
removed an unused import from that file earlier - so the replacement matched
nothing and the script cheerfully reported success.

**Fix.** Every patch script now asserts its anchor was found and exits non-zero
otherwise. The same mistake recurred once more, in a script written to fix this
very problem, which is a fair indication of how easy it is to make.

**The honest process note:** I changed a file, assumed the edit applied, and it
had not. The only reason I caught it was reading the output rather than trusting
the exit code. Verification has to be on the *artifact*, not on the tool that
produced it - which is the same lesson as 6.9, one level up.

### 6.11 The city resolved as "Now"

**Symptom.** In a three-turn demo run, the query "Now tell me about Kyoto"
resolved the city as **"Now"**, routed it to web search, and produced a complete,
confident answer about a city that does not exist.

**Cause.** The extractor tried the gazetteer, then fell back to "the first
capitalised word that is not sentence filler". "Now" is capitalised at the start
of a sentence and was not in my filler list.

**Fix.** Read the grammar instead of the capitalisation. A preposition names its
object, so "about Kyoto" / "in Osaka" / "to Lisbon" is checked first, and the
capitalised-word fallback now prefers a candidate that is *not* the opening word
of the sentence, because an initial capital carries no information.

**Why this one was worth chasing.** It is a *silent* failure: nothing raises, the
graph runs perfectly, every tool succeeds, and the user gets a beautifully
rendered answer about the wrong place. It was found by running the demo script
end to end and reading the output, not by any test - which is why the fix arrived
with eight regression tests covering the phrasings a panel is likely to type.

---

### 6.12 My stale threshold was rescued by the layer I added for reviewers

**Symptom.** The first screenshot run showed the trace panel reporting "similarity
0.207 against a threshold of **0.55**", with Tokyo marked as *not* above the
threshold - yet the answer came from the vector store anyway and looked perfect.

**Cause.** My own `.env` still carried the old 0.55 value from the plan, months
of reasoning ago in project time. Every routing decision was being made by the
gazetteer's exact-name layer, because no city can score 0.55 under the corrected
scoring. The similarity path was completely dead.

**Both halves of this matter, and they pull in opposite directions.**

The defence-in-depth I added for reviewers ended up covering my own
misconfiguration. That is the strongest possible argument for the layered router:
a bare threshold would have sent *every* query to web search and the demo would
have been visibly wrong. Layering meant the system stayed correct while one of
its two mechanisms was disabled.

It is also a warning, and the more useful half. **The redundancy masked a broken
setting.** Nothing failed, nothing logged an error, and the app behaved exactly
as intended - so the bug survived until I read the trace panel and noticed a
number that should not have been there. Redundancy buys correctness at the price
of silence, and silence is how misconfiguration persists.

**Fix.** The threshold guard already existed; it was only writing to the log. It
now renders as a banner on the page itself, so the next person cannot miss what I
nearly missed. The sidebar slider also widens its range rather than raising when
the configured value is out of bounds.

### 6.13 My own failures cluster around assuming an operation applied

Three separate bugs in this build share one root: I performed an operation, did
not verify it landed, and carried on. A patch script that printed "patched" while
matching nothing - twice, the second time in a script written to fix the first.
A `mkdir` placed outside the `try` block that was supposed to make storage
failures survivable. An edit whose anchor had silently moved because a linter had
reformatted the file underneath it.

The fix in each case was the same shape: assert the postcondition rather than
trusting the operation. Patch scripts now exit non-zero when an anchor is
missing, and verification is done on the artifact - the rendered diagram, the
generated file - not on the exit code of the tool that produced it.

---

### 6.14 A real API failure arrived unannounced, and the banner held

**Symptom.** During an automated screenshot run against the live provider, the
Kyoto page rendered a red banner reading:

    The request failed: APIConnectionError: Connection error.

**Cause.** A genuine transient network failure between my machine and Groq,
mid-capture. Not injected, not simulated - the real thing, and it also explains
why that run took ten minutes: the retry policy was doing its job with backoff
before finally giving up.

**Why I am recording it as a finding rather than an annoyance.** Every failure
path in this project up to that point had been exercised by *my own* injection
switches: a simulated 500, a simulated timeout, a simulated 429. Those prove the
handling works against the failures I imagined. This was the first time real
infrastructure failed on its own, and the result was a clean banner, an intact
page, and an app that still responded to the next request. The error handling
works on real failures, not only on the ones I designed.

It is the same lesson as section 13b from the other direction: I had tested my
assumptions thoroughly, and reality supplied the confirmation for free.

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

Those are the figures from one labelled run, and they are not a constant. The
mock providers apply 15 percent latency jitter, so repeating the measurement
gives a mean between roughly 1.9x and 2.1x, with individual queries ranging
from about 1.6x to 2.8x. The shape is stable; the exact number is not, and I
would quote it as "about 2x" rather than to two decimal places.

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

## 11. Step 11 - Distinction 3: memory and the follow-up

### 11.1 What a checkpointer is, in plain English

After every superstep, LangGraph hands the current state to a **checkpointer**,
which saves it under a **thread id**. When a later request arrives on the same
thread id, the graph resumes from that saved state instead of starting empty.

That is the entire memory system. There is no separate conversation store and no
summarisation step: the typed state *is* the memory, and the checkpointer is
where it lives between turns. When turn two asks "what about next week?", the
city is not re-derived from the text - it is simply still there.

**What `thread_id` scopes** is one conversation. Two browser tabs with different
thread ids are two conversations that cannot see each other's cities or results.
A test asserts a follow-up on an unseen thread finds nothing to borrow.

### 11.2 Why re-running and discarding would be cheating

This is the easiest of the three distinctions to fake. A graph could re-run every
branch on the follow-up, throw away the images it just fetched, and produce
user-visible behaviour identical to a graph that genuinely skipped them. The
answer would look the same. The API bill would not.

So the implementation makes the skip a property of the **topology**, and the
evidence auditable:

* `planned_branches(state)` returns what runs; `skipped_branches(state)` returns
  what does not. The fan-out edge dispatches the first list, the planning node
  reports the second, and a test asserts they are complementary - no branch can
  fall through the gap or appear in both.
* `timings` is reset at the start of every turn, so its keys are a precise record
  of what executed *this* turn. The test asserts `execute_images` and
  `retrieve_vector` are **absent** from turn two. A re-run-and-discard
  implementation would still have their keys and would fail.
* `skipped_nodes` and `skipped_ms_saved` are written into state, so the UI can
  show what was skipped and what it was worth.

### 11.3 What skipping actually buys, measured honestly

The tempting claim is "the follow-up is much faster". Measuring it made clear
that overstates the case. Three consecutive turn-1/turn-2 pairs:

| Turn 1 | Turn 2 | Wall-clock saved | Provider work avoided |
|---|---|---|---|
| 1279 ms | 835 ms | 444 ms | 1258 ms |
| 1219 ms | 822 ms | 397 ms | 1210 ms |
| 1093 ms | 1044 ms | 49 ms | 1082 ms |

**Mean wall clock saved: ~296 ms. Mean provider work avoided: ~1184 ms.**

Again, one run. Repeated, the wall-clock saving swings widely - between about
50 ms and 450 ms - because it depends on how the skipped branches happened to
overlap with the weather branch on the first turn. The work avoided is far
steadier, at roughly 1.1 to 1.3 seconds, because it is a sum of real provider
durations rather than a difference between two noisy wall clocks. That the
stable number is the one measuring work, and the noisy one is the one measuring
time, is itself the argument of this section.

The gap between those two numbers is the interesting part, and it follows
directly from Distinction 2. The skipped branches were running *concurrently*
with the weather branch on turn one, so removing them barely shortens the turn -
it still costs whatever the weather branch costs. What is genuinely saved is
**work**: an entire image-provider round-trip and a knowledge read whose results
would have been thrown away, plus the quota and money they cost against a live
API, and one fewer tool call for the model to plan.

So the honest framing, and the one I would give a panel: *parallelism buys
latency; skipping buys cost.* They are different wins, and the second is the one
that scales - at a thousand users, avoiding a redundant image fetch on every
follow-up matters far more than 300 ms.

The tests assert accordingly: the strong assertion is on work avoided, and the
weak one is only that the follow-up is not somehow *slower* than the turn that
did strictly more. An earlier version asserted a 15% wall-clock improvement and
was both flaky and dishonest.

### 11.4 The guarded failure: a follow-up with no history

A fresh thread asking "what about next week?" has no city in the question and
none in memory. The graph routes to `clarify` and answers with a question rather
than guessing.

Guessing would be strictly worse. A confident, complete answer about a city the
user never mentioned is harder to detect than a request for clarification - the
page looks entirely normal. The response object leaves `city` **empty** rather
than inventing a placeholder, and exposes `is_clarification` so the UI can render
it as a prompt rather than an answer.

### 11.5 MemorySaver versus SQLite, and a verified detail

`MemorySaver` is the default: no setup, fast, and adequate for a demo. Its
limitation is worth saying out loud rather than hiding - **it dies with the
process**. Restart the app and every conversation is gone.

`CHECKPOINTER=sqlite` swaps in a durable file-backed saver. The test that proves
this is real closes the first graph, its checkpointer and its database connection
entirely, builds a second graph from scratch, and asserts the follow-up still
resolves the city. Only the file on disk connects them, which is what
distinguishes genuine persistence from a process-local dictionary that happens to
survive two calls in one test.

One detail I verified rather than assumed: **the synchronous `SqliteSaver` cannot
be used here.** This graph runs through `ainvoke`, and the sync saver raises
`NotImplementedError: The SqliteSaver does not support async methods` on the first
checkpoint write. `AsyncSqliteSaver` is the one that works, and because its
`from_conn_string` is an async *context manager*, the connection is owned by the
checkpointer module so the saver can outlive a single `async with` block.

A related bug the tests caught: the directory creation for the database sat
*outside* the try block, so an unusable path crashed start-up instead of degrading
to the in-memory saver. Losing durability is survivable; failing to start is not.

### 11.6 The three-turn demo, end to end

```
TURN 1  "Tell me about Tokyo"        1062 ms   intent=new_city
        ran: classify_intent, plan_tools, retrieve_vector,
             execute_weather, execute_images, synthesize

TURN 2  "what about next week?"       984 ms   intent=weather_only
        ran     : classify_intent, plan_tools, execute_weather, synthesize
        skipped : retrieve_vector, execute_images  (1041 ms of work avoided)
        city    : Tokyo, carried from checkpointed state
        forecast: 2026-08-19 -> 2026-08-26   (the window actually moved)
        images  : preserved from turn 1, not re-fetched

TURN 3  "Now tell me about Kyoto"    1220 ms   intent=new_city
        ran: classify_intent, plan_tools, web_search,
             execute_weather, execute_images, synthesize
        route: web (similarity 0.040, below the 0.07 threshold)
```

Turn three is also the turn that exposed the "Now" bug in section 6.11 - the
first version of that run resolved the city as "Now" and routed *that* to the web.

### 11.7 One thing that surprised me

The follow-up refreshed the forecast for **the same seven days**. The intent was
classified correctly, the date range moved, the label said "next week" - and the
weather tool was still called without a start date, because the planning brief
passed the *label* to the model but not the date itself. The window moved in the
prose and nowhere in the data.

It was caught by the one test that compared the actual first forecast date across
turns rather than trusting the label. The lesson is the same one as 6.9 from a
different angle: the thing to assert on is the data the user ends up seeing, not
the metadata that describes it.

---

## 12. Step 13 - The interface, and the async bridge

### 12.1 The problem the runner solves

Streamlit re-executes the whole script on every interaction. The graph is async.
The obvious bridge - `asyncio.run(app.ainvoke(...))` inside the script - builds a
new event loop per rerun and tears it down immediately, which breaks two things:
anything bound to a loop dies with it (the SQLite checkpointer holds an
`aiosqlite` connection created on a specific loop), and every rerun re-pays the
cost of constructing the loop, the providers and the graph.

So: **one event loop, running forever on a dedicated daemon thread, created once
per session.** Work is submitted with `run_coroutine_threadsafe` and waited on
with a timeout. The loop outlives every rerun.

### 12.2 The ownership trap

The loop, the compiled graph and the database connection must share one lifetime,
and the connection has to be created **on the loop that will later use it**.
Caching the loop while creating the connection per rerun is the pairing that
deadlocks: the connection belongs to a loop nobody is running, so the first
`await` never returns and the UI hangs with no error at all.

`AgentRuntime` owns all three, and `build_runtime` constructs them in that order.
Streamlit caches the whole object or none of it. A test drives a SQLite turn
followed by a follow-up specifically because a regression there would appear as a
hang rather than as a failure.

Sidebar toggles **mutate the runtime's live `Settings` object** instead of
rebuilding it. Rebuilding would tear down the loop and the conversation history,
so flipping "break the weather API" would silently reset the demo.

### 12.3 What the trace panel is for

It is the highest-leverage screen in the project. Without it a reviewer sees a
summary, some photographs and a chart, and has no way to tell whether the facts
came from the knowledge base or the web, whether anything ran concurrently, or
what a follow-up skipped. With it, each of those is legible in seconds - and
crucially it shows the *numbers*, not the verdicts:

> Source: Internal knowledge base. Decided by exact name match on 'Tokyo'.
> Similarity 0.207 against a threshold of 0.07.

"Routed to the vector store" is a claim. That sentence is an explanation.

The Memory tab states the follow-up saving in the honest terms from section 11.3 -
work and API cost avoided, not wall clock - because a UI that overstates its own
cleverness is worse than one that says nothing.

---

## 13. Step 12 - Structured output, grounding, and the live run

### 13.1 What the model is and is not asked for

The model writes prose. It is **not** asked for the forecast or the image URLs,
even though both appear in the final object. Those are typed values the tools
already returned, and asking a model to copy them back would create an
opportunity to alter them for no benefit. The schema it fills in has two fields:
`city_summary` and `highlights`.

Validation is JSON mode plus this project's own Pydantic pass, not
`with_structured_output`. Not every provider supports schema-constrained
decoding, the ones that do implement it differently, and keeping the failure
handling in code I can explain is worth more here than a framework helper.

The repair path is: validate, and on failure hand the model its own error text
once. If that also fails, assemble the response deterministically from the tool
payloads. **The user never sees a validation error** - the worst case is a duller
summary.

### 13.2 Grounding matters more than prose quality

The "Now tell me about Kyoto" bug produced a complete, confident, well-written
answer about a city that does not exist. That is the canonical failure mode of
these systems, and prose quality actively works against you: the better it reads,
the more convincing the mistake.

So the prompt gives the model the retrieved passages and instructs it to use
nothing else; sparse context is flagged explicitly so the correct output becomes
"there is limited information available" rather than fluent invention. A test
runs the synthesizer with an empty corpus and asserts the summary says exactly
that, and another asserts every sentence in the deterministic fallback appears
verbatim in the source material.

### 13.3 The live run, and what only a live provider could reveal

One deliberate request against Groq, `openai/gpt-oss-120b`:

```
route          : vector (exact, score 0.207)
tools executed : search_city_images, get_weather_forecast
forecast points: 7
images         : 4
validated      : TravelResponse passed Pydantic validation
tokens         : 2802 (1744 prompt + 1058 completion) across 2 calls
parallel       : 2827 ms sequential-equivalent vs 1900 ms actual (1.49x)
```

The summary it wrote is grounded in the seeded corpus - Yamanote, Kabukicho,
Toyosu, Suica and Pasmo, the Narita Express timing - and it correctly folded in
the *tool's* weather payload ("24 to 31 C over the next week") rather than
inventing a forecast.

**But the first live run was different, and this is the point of doing it.** The
model called only the weather tool. The image branch had nothing to execute and
the gallery came back empty. My mock always requested both tools, so nothing in
293 passing tests could have caught it.

Strengthening the prompt did not fix it. What fixed it was accepting that the
interface has a fixed contract - a summary, a gallery and a chart - and that
whether the page needs photographs is not really the model's judgement call. The
graph now completes a plan that omits a required tool, synthesising the missing
call with sensible arguments and recording it in the trace as
`tools_added_by_graph`.

The trade-off, stated plainly: **the planner is now advisory for tool selection
rather than authoritative.** That is the right split when the required set is
known up front. If the tool set were open-ended it would be the wrong design, and
the prompt would have to carry the weight instead.

Two smaller things the live run exposed: the model emits Unicode the Windows
cp1252 console cannot encode (U+202F, a narrow no-break space), which killed a
script *after* its work had succeeded; and a later run hit a genuine
`APIConnectionError` mid-capture, which the UI rendered as a clean banner with the
app still usable. The second was an unplanned but welcome demonstration that the
error handling works on real failures, not only simulated ones.

---

## 13b. My mocks were more obedient than reality

This is the most valuable thing I learned building this project, so it gets its
own section.

### The finding

At the point I ran the first live request against Groq, the suite was at 293
passing tests. Every path was covered: routing both ways, the manual executor,
the fan-out, the follow-up skip, graceful degradation, the UI. I had good reason
to think the system worked.

The live run returned a page with **no images at all**.

The model - `openai/gpt-oss-120b`, offered both the weather tool and the image
tool - had called only the weather tool. The image branch had nothing to execute,
so it recorded a skip and moved on. Nothing failed. No test could have failed,
because my `MockLLM` *always* requested both tools.

That is the whole lesson in one sentence: **my mocks were more obedient than
reality.** I wrote a mock that did what the system needed, then tested that the
system worked when the model did what it needed. The 293 tests were structurally
incapable of discovering a model that simply decided pictures were unnecessary.

### Why prompt strengthening was the wrong fix

My first instinct was to fix the prompt. I rewrote the planner instruction to say
a complete answer always contains a summary, a gallery and a chart, and that
omitting a tool leaves a visibly empty panel in the interface.

The model called only the weather tool again.

I could have escalated - stronger wording, few-shot examples, a reminder in the
user turn. But that approach was wrong even when it looked like it might work,
because **it makes a hard contract depend on persuasion.** The interface needs
four image URLs to render its gallery. That requirement does not vary with
temperature, model version, phrasing, or how the model happens to feel about a
particular city. Encoding it in a prompt means encoding it in the one part of the
system with no guarantees at all, and every future model upgrade re-rolls the
dice.

### The actual fix

The UI has a fixed contract: a summary, a gallery and a chart. Because the
required tool set is therefore known *before* the model is asked anything, the
graph can check the plan against it. `_complete_plan` compares the tools the
model requested with the tools it was offered, synthesises any that are missing
with sensible arguments, and records what it added in the trace as
`tools_added_by_graph`.

The model still chooses the arguments. It still drives the routing. It simply no
longer gets to decide whether the page needs its gallery.

### The trade-off, stated plainly

**The planner is now advisory for tool selection rather than authoritative.**

That is the correct split *here*, because the required set is known in advance
and is small. It would be the wrong design if the tool set were open-ended - if
the agent could choose among fifty tools depending on the request, there would be
no "required set" to complete a plan against, and the prompt would have to carry
the weight after all. The design is right for this problem, not in general, and I
would say so before defending it.

It is also disclosed rather than hidden: the trace panel shows exactly which
calls the graph added, so nobody reading the output mistakes the graph's decision
for the model's.

### The general lesson

**Mocks verify your code against your assumptions, not against reality.** They
are excellent at catching regressions in logic you have already understood, and
useless at catching the thing you did not think of - because you wrote them, out
of the same understanding that produced the bug.

That is precisely why the live smoke run exists, and the honest criticism is that
**it should have come earlier.** I built it as a final check before the demo. If
I had run one live request straight after the manual executor landed, I would
have found this on day one instead of after the UI was finished. On any future
project of this shape I would put a single live call in the loop as soon as the
tool protocol works, and keep everything else on mocks.

---

## 14. Requirement traceability

| Assignment requirement | Where it lives | One-line explanation |
|---|---|---|
| Vector store seeded with 3 cities | `data/city_facts/*.md`, `scripts/seed_vectorstore.py` | 27 chunks across Paris, Tokyo and New York, 9 each. |
| Typed state | `src/travel_agent/schemas/state.py` | `TypedDict` with `Annotated` reducers on every concurrently-written key. |
| Structured output object | `src/travel_agent/schemas/response.py` | `TravelResponse` with `city_summary`, `weather_forecast`, `image_urls`. |
| Manual tool execution (Distinction 1) | `graph/nodes/tool_executor.py` | Hand-parses `tool_calls`, validates against the Pydantic schema, dispatches, and returns `ToolMessage` with matching ids and `status="error"` on failure. No prebuilt helpers - enforced by a test. |
| Parallel fan-out (Distinction 2) | `graph/edges.py::route_and_fan_out`, `graph/nodes/core.py::join` | A conditional edge returns a list of node names, so all branches run in one superstep. Measured mean speed-up 2.03x. |
| Checkpointer + follow-up (Distinction 3) | `graph/checkpointer.py`, `graph/edges.py::planned_branches` | MemorySaver by default, AsyncSqliteSaver when durable; a follow-up re-runs only the weather branch and records what it skipped. |
| Conditional edge on knowledge availability | `services/router.py`, `graph/edges.py::route_and_fan_out` | Layered decision - gazetteer first, then centroid similarity against the threshold - wired as the second conditional edge. |
| Web search path for unknown cities | `tools/search/{mock,live}.py` | Mock plus DuckDuckGo and Tavily implementations. |
| Weather tool, 5-7 day forecast | `tools/weather/{mock,live}.py` | Climate-plausible mock; OpenWeatherMap live. |
| Image retrieval | `tools/images/{mock,live}.py` | Verified Commons photographs; Unsplash live. |
| Graceful degradation when a tool fails | `tools/registry.py`, `tools/retry.py` | `execute()` returns an error result rather than raising; retries are bounded. |
| Streamlit UI with chart and gallery | `ui/app.py`, `ui/components/*` | Plotly forecast chart with min/max bands, image gallery, and a live agent-trace panel. |
| `graph.png` | repo root, `scripts/export_graph.py` | Generated from the compiled graph, labelled to distinguish XOR from concurrent edges, and committed so it exists offline. |

---

## 15. The full request lifecycle

What happens between typing "Tell me about Kyoto" and pixels on screen. Kyoto is
the interesting case because it is *not* in the knowledge base.

1. **You press Send.** Streamlit re-runs the whole script. `ui/app.py` reads the
   cached `AgentRuntime` - the event loop, the compiled graph and the database
   connection built once for this session - rather than constructing anything.
2. **The query is submitted to the loop.** `runtime.invoke` calls
   `asyncio.run_coroutine_threadsafe(app.ainvoke(...), loop)` and waits with a
   timeout. The UI thread blocks on a future; the graph runs on its own thread.
3. **`normalize_input`** tidies the text, increments the turn counter, and clears
   the per-turn observability keys by passing `None` to their reducers. Slots and
   previous results are deliberately left alone.
4. **`classify_intent`** extracts the city. The gazetteer finds no match for
   "Kyoto", so the grammatical extractor takes over: the preposition in "about
   Kyoto" names its object. No date language is present, so the window stays at
   seven days from today. City changed, so the intent is `new_city`.
5. **The first conditional edge** (`route_after_intent`) sees real work to do and
   routes to `plan_tools`.
6. **`plan_tools` decides where the facts come from.** The router tries the
   gazetteer first - no match - then scores "Kyoto" against each city profile.
   The best is 0.040 against Tokyo, below the 0.07 threshold, so the route is
   `web`. The score, the threshold, the reason and every city's score go into
   state for the trace panel.
7. **The same node asks the model which tools to call**, offering the weather
   tool, the image tool and - because the route is `web` - the web search tool.
   The model replies with an `AIMessage` carrying a `tool_calls` payload.
   `_complete_plan` checks nothing required is missing and adds it if so.
8. **The second conditional edge** (`route_and_fan_out`) returns a *list*:
   `["web_search", "execute_weather", "execute_images"]`. All three are scheduled
   into one superstep and run concurrently.
9. **Three branches execute at once.** Each tool branch is the same
   `ManualToolExecutor` class with a different `handles` set: it reads
   `messages[-1].tool_calls`, picks out the calls it owns, validates the arguments
   against that tool's Pydantic schema, dispatches through the registry - which
   applies the timeout, the bounded retry and the backoff - and appends a
   `ToolMessage` carrying the matching `tool_call_id`. A failure becomes a
   `ToolMessage` with `status="error"` rather than an exception.
10. **`join` runs once all three have finished**, because that is what the next
    superstep means. It compares the sum of the branch durations against the
    superstep's wall clock and stores both numbers plus the ratio.
11. **`synthesize`** builds a prompt containing only the passages actually in
    state, asks the model for JSON, validates it against `SynthesisDraft`,
    repairs once if that fails, and falls back to a deterministic summary if the
    repair fails too. The forecast and image URLs are copied from the typed tool
    payloads, never from the model.
12. **The state is checkpointed** under the thread id, so the next turn can
    resume from it.
13. **The future completes** and the UI thread wakes with the final state.
14. **The page renders** from the validated `TravelResponse`: heading, source
    line, warnings, summary, highlights, gallery, Plotly chart, sources. The trace
    panel renders from the same state - route, scores, tool timings, the parallel
    measurement, what was skipped, the start-up checks.

Measured end to end on mocks: about 1.0 to 1.3 seconds. On the live Groq path,
about 5 seconds, dominated by the two model calls.

---

## 16. Things I deliberately did not do

* **No `sentence-transformers` by default** - hundreds of megabytes of PyTorch and
  a slow cold start for a three-city corpus. The interface supports it, and
  `EMBEDDING_PROVIDER=openai` gives real semantic embeddings for anyone who wants
  them.
* **No `langchain-community`** - the LangChain 1.x package split makes it a
  version-drift risk and nothing here needs it.
* **No `tenacity`** - a 30-line retry utility I can explain line by line is worth
  more than a dependency here, and it let me implement `Retry-After` handling
  exactly the way Groq's rate limiting needs.
* **No dollar-cost display** - the three supported providers price differently, so
  the UI shows token counts and names the provider rather than printing a figure
  that would be wrong for two of them.
* **No streaming output** - it would improve the live experience materially, but it
  complicates the structured-output contract, and a validated object was the
  requirement.
* **No authentication or multi-user isolation** - `thread_id` separates
  conversations, but anyone can supply any thread id. That is correct for a local
  demo and inadequate for anything deployed.
* **No caching layer** - the obvious next optimisation, and deliberately left out
  so the parallel and skip measurements reflect real work rather than cache hits.
* **No live weather or image keys used in the demo** - the mocks are the default
  because the assignment blesses them and because a demo that depends on a
  reviewer's API keys is a demo that fails in the room. Both live providers are
  implemented and are one env var away.

---

## 17. Interview questions and answers

Twenty questions I expect, with the answers I would give.

### Architecture

**1. Why LangGraph rather than a plain chain?**
Because this system makes decisions and does work concurrently, and a chain can
express neither. Three things needed a graph: a conditional edge that picks the
knowledge source at runtime; a fan-out where three branches run in one step and
rejoin; and a follow-up turn that skips two of those branches entirely. In a
chain, all three become `if` statements buried inside application code, invisible
in any diagram. The trade-off is real - more machinery, and every state key
written concurrently now needs a reducer - but the topology becomes the
documentation.

**2. What is a superstep?**
LangGraph executes in rounds rather than one node at a time. Everything scheduled
into the same round starts together, runs concurrently, and the next round does
not begin until all of it finishes. That is why my `join` node needs no
synchronisation code: it is simply the node after the round, so it cannot run
early.

**3. Why are the reducers mandatory rather than stylistic?**
When two nodes in one superstep write the same state key, LangGraph does not pick
a winner - it raises `InvalidUpdateError: Can receive only one value per step`. A
reducer attached via `Annotated` tells it how to merge. So the fan-out is only
legal because the reducers exist. I have two tests pinning that: one builds a
fan-out over an un-reduced key and asserts it raises, and a control test adds the
annotation to the identical topology and asserts it succeeds.

**4. Walk me through your state design.**
One `TypedDict` with about twenty keys in four groups: the conversation, the
resolved slots, the routing decision, and observability. Keys written by
concurrent branches carry reducers - `append_list` for the trace and errors,
`merge_timings` for durations, `add_token_usage` for usage. Keys with a single
writer per turn use `replace_value`, which exists so a follow-up turn does not
wipe the previous turn's images. The accumulating reducers treat an explicit
`None` as "reset", which is how each turn starts with a clean trace.

### The three distinctions

**5. Why not use `ToolNode`?**
Beyond the assignment asking for it, writing the executor by hand bought three
things. Per-tool error isolation: each call runs independently, so one dead tool
becomes one error message while its siblings return data. Selective execution:
one executor instance can be told to handle only some of the tool calls, which is
what lets the weather and image branches share the same class while running
concurrently in different nodes. And observability: every call emits a trace
event with the tool, id, arguments, provider, attempts and duration. What I gave
up is that I now own schema validation and id correctness - both silent failures
if I get them wrong - which is why those are the most heavily tested parts.

**6. Explain the tool-calling protocol.**
Three parts. I advertise tools as JSON schemas generated from Pydantic models.
The model replies not with prose but with a structured request: an id, a tool
name, and arguments. I run the function and reply with a `ToolMessage` carrying
the *exact* id from the request. The id matters because a model can request
several tools at once and my replies come back in whatever order the tools
finish - frequently not the request order, since they run concurrently. Get the
id wrong and the model attributes the weather data to the image request and
nothing raises. Omit a reply entirely and most providers reject the next request
outright.

**7. Why does `status="error"` matter?**
If a tool fails and I return "error: connection timed out" as an ordinary result,
the model reads that as the legitimate output and summarises "the weather is
error: connection timed out". `ToolMessage.status="error"` tells it the call did
not succeed, so it can work around the gap honestly. I found the field by
introspecting the installed class during the Phase 1 recon rather than assuming
the shape.

**8. How does your parallelism actually work, and what bounds it?**
A conditional edge returns a *list* of node names, which schedules them all into
one superstep. Measured across three queries: 1.70x, 1.61x and 2.77x, mean
**2.03x**, saving roughly 1.1 seconds per request. What bounds it is the slowest
branch - concurrent work costs the maximum, not the sum. On the vector-store path
the local index read is effectively free, so the fan-out is really weather
against images and the ceiling is the slower of the two. The web path gains most
because it has three genuinely slow branches.

**9. Why not just `asyncio.gather` inside one node?**
It would run the same work concurrently, but the graph would contain one node
where three should be, and `graph.png` - a required artifact - would show a
straight line. Parallelism a reviewer cannot see in the topology is parallelism
they have to take on faith. It would also mean hand-rolling failure isolation and
partial-state merging, and per-node timings would collapse into one number so the
trace panel could not show which branch was slow.

**10. What does the follow-up skip actually save?**
Not much latency, and I would correct anyone who assumed otherwise. Measured over
three turn-pairs: mean wall-clock saving **296 ms**, mean provider work avoided
**1184 ms**. They diverge because the skipped branches previously ran
*concurrently* with the weather branch, so removing them barely shortens the
turn. The saving is an image-provider round-trip and a knowledge read that would
have been discarded, plus the quota and money they cost. **Parallelism buys
latency; skipping buys cost.** The second is what scales - at a thousand users,
avoiding a redundant fetch on every follow-up matters far more than 300 ms.

**11. How do I know you skip rather than re-run and discard?**
Because `timings` is reset at the start of every turn, so its keys are a precise
record of what executed *this* turn, and the test asserts `execute_images` and
`retrieve_vector` are **absent** on turn two. A re-run-and-discard implementation
would still have their keys. The graph also records `skipped_nodes` and
`skipped_ms_saved` in state, and the UI displays them.

**12. What is a checkpointer, and what does `thread_id` scope?**
After every superstep LangGraph hands the state to the checkpointer, which files
it under a thread id; a later turn on that id resumes from it. The typed state
*is* the memory - there is no separate conversation store. `thread_id` scopes one
conversation, and a test asserts two threads cannot see each other's cities.
`MemorySaver` is the default and dies with the process; `CHECKPOINTER=sqlite`
gives durability, proven by a test that closes the graph, the checkpointer and the
connection, rebuilds from the file alone, and asserts the follow-up still resolves
the city.

### Retrieval and routing

**13. How does routing decide, and where did 0.07 come from?**
Layered. First the gazetteer: does this name, or an alias, match a city I hold
documents for? That is a question about names and a lookup answers it exactly.
Only if that fails does the similarity score decide - the query is embedded and
compared against each city's profile vector. 0.07 came from measurement, not
intuition: seeded cities score 0.102 to 0.207, unseeded ones 0.000 to 0.040, so
the separation gap is 0.062 and 0.07 sits inside it. The seeder prints that
matrix on every run.

**14. Why layered rather than a bare threshold?**
Because a bare threshold is one fragile gate. "NYC" is a city I hold nine
documents about, and it is not a token in the corpus at all - it would fail on
score alone. Layering also proved itself accidentally: my own `.env` carried a
stale 0.55 threshold for a while, which no city can reach, and every request was
being routed correctly by the gazetteer while the similarity path was dead. That
is the argument for defence in depth and the warning about it in one incident -
the redundancy kept the system correct and hid a broken setting until I read the
trace panel.

**15. How would you know if the router were wrong?**
Three ways, in increasing rigour. The trace panel shows the score and threshold on
every request, so a wrong decision is visible rather than silent. The start-up
guard measures the separation and refuses to stay quiet if the threshold sits
outside it. And `evals/queries.jsonl` holds twenty labelled queries - seeded
cities, unseeded cities, aliases, no-city cases and the phrasings that caused real
bugs - currently 20/20. That last one is what I would grow first: routing accuracy
is an empirical question and deserves a number, not an opinion.

**16. Why a hashed TF-IDF embedder rather than a real model?**
Because the retrieval problem here is narrow - decide whether a query names one of
three seeded cities - and that is lexical, not semantic. The alternative costs
hundreds of megabytes of PyTorch or an API key, and the project promises to run
with neither. The trade-off is real and I would state it unprompted: this embedder
does not know "the French capital" means Paris. `EMBEDDING_PROVIDER=openai` swaps
in real semantic embeddings without touching the store or the router.

### Failure, operations and scale

**17. What happens when the weather API dies?**
The page still renders. The registry never raises - it returns a result object
carrying either a payload or an error - so the executor turns the failure into a
`ToolMessage` with `status="error"`, the other branches complete normally, and
`synthesize` is told the forecast is unavailable so it says so rather than
inventing one. The user sees a warning banner naming the actual failure. There is
a sidebar toggle that breaks it on demand, in four different ways, so this can be
demonstrated rather than described.

**18. Why do you not retry a malformed response?**
Because retrying only helps when a failure is transient. A timeout, a 500 and a
429 are all worth another attempt. A malformed payload is not: the provider
answered successfully with data I cannot parse, so the identical request produces
the identical unusable response. Retrying it three times spends triple the user's
time and triple the quota to fail the same way - and on a rate-limited free tier
those wasted calls can push the next legitimate request over the limit. The
decision lives in the exception hierarchy: `MalformedPayloadError` deliberately
does not inherit from `RetryableError`.

**19. Why Groq when the assignment named OpenAI and Anthropic?**
Both named providers are fully implemented behind the same interface and covered
by the same tests; switching is one line in `.env`. Groq is the demo driver for
two practical reasons: the free-tier allowance on the others does not comfortably
cover a day of iterative development plus a live demo, and Groq is fast enough
that the parallel measurement is dominated by tool latency rather than model
latency, which makes the number cleaner. It is also OpenAI-compatible and returns
a genuine `tool_calls` payload, so the manual executor is exercised against the
real protocol.

**20. How would you scale this to a thousand users, and what breaks first?**
The graph itself is stateless per request, so it scales horizontally - but three
things break in order. First, `MemorySaver`: it is per-process, so any load
balancing loses conversations. That is a config change to the SQLite saver, or
Postgres in production. Second, the vector store: it is loaded into each process,
which is fine at 27 chunks and wrong at scale - that becomes a shared service.
Third, provider rate limits, which is the real ceiling: at a thousand users the
weather and image calls dominate, so I would add a shared cache keyed by city and
day, which would serve most traffic without a provider call at all. The retrieval
being in-process is the thing I would fix first, because it is the one that
silently multiplies memory per worker.

**21. What are the security concerns?**
Four. Prompt injection through retrieved content - the web search path feeds
third-party text into a model prompt, and while the current tools are read-only,
that becomes serious the moment a tool can act. Key handling: keys live only in
`.env`, which is gitignored, and never reach logs or the UI. Untrusted URLs: the
image URLs are rendered in a browser, so the response schema validates that each
is http(s) and drops anything else, and the curated set is a fixed allow-list.
And resource exhaustion: every external call has a timeout and a bounded retry, so
a hanging provider cannot pin a worker indefinitely.

**22. What would you build next?**
In order. A proper eval harness around the labelled query set, with the threshold
swept rather than fixed. A shared cache keyed by city and day, since the same
three cities dominate traffic and the data changes daily at most. Streaming the
summary token by token, because five seconds of silence is the weakest part of the
live experience. And LangSmith tracing, because my trace panel shows one request
well and nothing about the hundred before it.

---

