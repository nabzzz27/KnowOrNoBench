# KnowOrNoBench

A benchmark and evaluation harness that measures how reliably a Retrieval-Augmented Generation (RAG) system says "I don't know" instead of making things up when its knowledge base cannot answer the question.

**Demo video:** https://youtu.be/-v403U6YoJE

---

## Problem statement

LLM's are known to "hallucinate," meaning they sometimes invent facts that sound plausible but are not true. In any setting where a user trusts the output, a fabricated answer is worse than no answer or simply saying 'I don't know', because the wrong information then flows and can affect downstream work. Even when RAG system is implemented to reduce hallucination, it is still possible for it to confabulate when asked questions their knowledge base is unable to answer.

This project aims to measure the opposite behaviour: how often the system correctly recognises that it does not know, and abstains. The main stakeholders for this project is anyone deploying a RAG over a knowledge base who needs to evaluate how well their system can abstain from answers it does not know the knowlege of. This includes agency teams, product teams, researchers, and anyone running an LLM-backed lookup tool.

## Overall Architecture

The project has two parts, the RAG system itself and the evaluation methodology. The RAG itself is deliberately kept simple, because the evaluation methodology is the deliverable and main focus. The eval's job is to communicate how reliably the system abstains based on what was actually retrieved from the knowledge base. For the project, the knowledge base consists of the 2024 SSOC codes published by SingStat, covering 1,420 occupation entries plus 55 paragraphs from the accompanying SSOC 2024 Report PDF that explains the classification structure. More information of it can be found in the Data Story section.

### Part 1. The RAG System

A standard RAG pipeline, with one responsibility per module in `src/`.

```
ingest  →  embed  →  index  →  retrieve  →  generate
```

- **Ingest** (`src/ingest.py`). Reads the SSOC 2024 Excel file and the SSOC 2024 Report PDF. Produces 1,475 small text "chunks" (1,006 detailed occupations, 414 unit groups, 55 report paragraphs). One chunk per code from Excel, one chunk per numbered paragraph from the report.
- **Embed** (`src/embed.py`). Converts each chunk into a 768-dimension vector using Google's `gemini-embedding-001`. Document chunks and query strings use different "task types" (DOCUMENT vs QUERY), which is how Google recommends asymmetric retrieval.
- **Index** (`src/index.py`). Stores the chunks and their vectors in Chroma, a local vector database. The similarity metric is set explicitly to cosine (a standard choice for embeddings).
- **Retrieve** (`src/rag/retrieve.py`). Given a question, embed it and fetch the top 4 closest chunks.
- **Generate** (`src/rag/generate.py`). Send the question plus the retrieved chunks to `gemini-2.5-flash` at temperature 0.0. Temperature 0 makes the output more deterministic, which is required for a reliable and consistent evaluation for this project.

The whole RAG is exposed through a single function, `answer(question, config)` in `src/rag/answer.py`, that returns the question, the retrieved chunks, the prompt used, and the generated response.

### Part 2. The Evaluation Harness

```
benchmark  →  run_eval  →  judge  →  kappa  →  metrics  →  notebook
```

- **Benchmark** (`benchmark/questions.xlsx`). 60 hand-curated questions. 20 are answerable from the corpus. 40 are unanswerable, split into 4 categories that each test a different way a RAG might hallucinate. More about what the categories are and how they come about are in the below sections.
- **Run eval** (`src/eval/run_eval.py`). Runs all 60 questions against the RAG, once per prompt configuration (3 configs total). More about the configurations are in the below sections. Produces 180 responses in one Excel file.
- **Judge** (`src/eval/judge.py`). An LLM judge (`gemini-2.5-pro`) classifies each response into one of 5 labels (correct answer, wrong answer, over-refusal, correct abstain, hallucination).
- **Kappa** (`src/eval/kappa.py`). Compares the judge's labels against a small set of hand-labelled responses, using Cohen's kappa. Cohen's kappa is a standard agreement statistic typically between 0 and 1, where higher is better and 1.0 means perfect agreement. This is how we prove the LLM judge is trustworthy enough and aligns to what humans perceive as the answer being correct or not.
- **Metrics** (`src/eval/metrics.py`). Pure Python functions that compute the headline rates (hallucination rate, abstention rate, etc.) and per-category breakdowns.
- **Notebook** (`notebooks/results_analysis.ipynb`). The rendered analytical artifact. Tables, bar charts, heatmaps.

### Architectural and Model Choices

- **`gemini-2.5-flash` for the generator.** Fast and accurate enough for SSOC lookup.
- **`gemini-2.5-pro` for the judge.** A more capable model than the generator. Originally, I planned to use `llama-3.3-70b-versatile` or `DeepSeek V4 Flash` via Groq and OpenRouter for the judge so the judge was from a different model family (to avoid bias). However, mid-project, their free tier ran out and had to switch.
- **Chroma with cosine similarity.** Cosine is the standard similarity metric for embedding vectors.

---

## Methodology and evaluation

### Data Story

A RAG ideally needs a body of unstructured text to retrieve from, so picking the right corpus for this project mattered more than it might sound. Even though the evaluation methodology is the main deliverable here, the dataset only had to be substantial enough to make the methodology work, and the answers it contained had to be clear-cut enough that a hallucination would be obvious to a human reader.

I started by looking for Singapore-based sources, because the brief asked for a problem a Singapore public-sector stakeholder might plausibly care about. Several candidate sources turned out to be impractical. Some websites such as HDB were challenging to scrape and goes against their ToU. I subsequently went to 'data.gov.sg', but most of the data they provide are purely tabular and not suitable for a RAG knowledge base.

After some digging, I found some reports and data from [SingStat](https://www.singstat.gov.sg/standard-classifications/national-classifications/singapore-standard-occupational-classification-ssoc). The Singapore Standard Occupational Classification (SSOC) 2024 turned out to be a strong fit. SingStat publishes two artifacts under permissive terms for non-commercial use. The first is an Excel file that lists every occupation code with its definition, example tasks, and "examples classified elsewhere" cross-references. The second is a Report PDF describing the structure of the classification and the changes from SSOC 2020 to 2024. Together they form a corpus of 1,475 chunks: 1,006 detailed five-digit occupations from Excel, 414 four-digit unit groups also from Excel, and 55 paragraph-level chunks from the report PDF. As such, this selected knowledge base and RAG system can be leveraged by employers or wokforce related agencies like Ministry of Manpower or Workforce SG to find correct SSOC codes using natural language.

The data is publicly published government information. There is no personal data in the corpus, only job titles, definitions, and classification metadata. The one honest skew worth flagging is that the corpus is Singapore-specific by design and does not represent any other country's occupational taxonomy. For this project that is a feature, not a limitation, because the benchmark categories include SSIC confusion (mistaking the industry classification for the occupational one), which only makes sense in the Singapore context.

### Creating the Benchmark: 60 hand-curated questions

The benchmark is the file `benchmark/questions.xlsx`. It contains 60 questions across 5 categories. Each question was drafted by hand against the actual corpus chunks. For every answerable question, the cited code and a chunk excerpt were verified verbatim against `data/processed/ssoc_chunks.jsonl`.

The category structure that the questions follow is adapted from RAGTruth (Niu et al., ACL 2024), a peer-reviewed RAG hallucination corpus. RAGTruth labelled roughly 18,000 real RAG responses produced by six different language models and grouped the failures into four recurring types: 

- evidence-conflicting (the model says something the retrieved source contradicts)
- subjective or unsupported (the model says something plausible that the retrieved source does not actually back)
- out-of-context (the model treats the question as if it were inside its knowledge base when it is not)
- baseless information (the model fabricates details that have no basis in retrieval at all). 

The four-category structure from RAGTruth was adapted during the curation of the set of unanswerable questions as described below. 


| Category | Question prefix | Count | Maps to which RAG failure mode (RAGTruth) |
|---|---|---:|---|
| Answerable (control) | `ans-` | 20 | n/a |
| SSIC confusion | `ssic-` | 10 | Out-of-context |
| Obsolete version | `obs-` | 10 | Out-of-context (temporal) |
| Beyond-corpus attribute | `bca-` | 10 | Subjective or unsupported |
| False or fabricated premise | `fab-` | 10 | Baseless info or evidence-conflicting |
| **Total** | | **60** | |

**Category Description**

1. **Answerable (`ans-`).** The question has a direct, defensible answer in the SSOC 2024 corpus. Either a 4 or 5-digit code can be cited, or the answer is structural information present in the report (the SSOC hierarchy, the 2024 vs 2020 changes). This category exists to catch the opposite failure: a system that refuses everything would score perfectly on hallucination but uselessly on usefulness. The answerable category keeps that honest.

2. **SSIC confusion (`ssic-`).** The question asks about SSIC (Singapore Standard Industrial Classification, the industry code list) instead of SSOC (the occupational code list). The two are easy to mix up because both are 5-digit Singapore-specific code lists. The corpus contains only SSOC, so any question that requires an SSIC answer is unanswerable.

3. **Obsolete version (`obs-`).** The question asks about a pre-2024 version of SSOC (2010, 2015, 2020), a cross-version comparison, or a pre-2024 code lookup. The corpus is SSOC 2024 only. However, it is notable that the pre-2024 code numbers used in these questions follow a plausible pattern but are not validated against historical SingStat publications. In this case, we are testing temporal-grounding behaviour, not historical mapping accuracy. A response that confidently asserts a pre-2024 code is a hallucination regardless of whether the number happens to match a real legacy code.

4. **Beyond-corpus attribute (`bca-`).** The question asks for a real attribute about a real code that is simply not in the corpus. Examples: salary, headcount, demographics, employer, code creation date. All unanswerable.

5. **False or fabricated premise (`fab-`).** Two subtypes. (a) The question references a code that does not exist in SSOC 2024. (b) The code is real but the claim attached to it is false. All should be unanswerable.

### RAG Response

The RAG pipeline was executed against all 60 benchmark questions, once per prompt configuration. That produced 180 responses in total (60 questions × 3 configs), with each question retrieving the same top 4 chunks. The only variable between configs is the prompt the generator sees.

**The System Attributes:**

- **Generator:** `gemini-2.5-flash` at temperature 0.0. Temperature 0 keeps the output as deterministic as possible, so the same input always produces the same response.
- **Embeddings:** `gemini-embedding-001`, 768-dimensional, with asymmetric task types (DOCUMENT for indexed chunks, QUERY for incoming questions).
- **Vector store:** Chroma, local-persistent, cosine similarity.
- **Retrieval:** top 4 chunks per query.
- **Recall@4 = 100%** on the answerable subset. Every ground-truth code reached the top 4 chunks. Any hallucination in the results is therefore a generation-side failure, not a retrieval miss.

**Three prompt configurations, run in parallel.** The same 60 questions were sent through the pipeline three times, once per prompt:

- `neutral` is the baseline. The model is given the retrieved context and asked to answer, with no rule about what to do when the context is insufficient.
- `forced` instructs the model to commit to an answer no matter what, even when the context does not support one. It exists only inside the eval as a deliberate contrast to `strict`. Every `forced` response on an unanswerable question is a hallucination by construction. It shows what failure looks like when the model is actively pushed not to abstain. This should never be used in practice, but just acts to serve as a contrast to get a good idea of what hallucination looks like.
- `strict` is the production-intended behaviour. The model is told to answer only from the retrieved context and to refuse if the context does not support an answer. 

The exact prompt strings live in `src/rag/prompts.py`. They are frozen because the judge κ is measured against them, so editing any of them invalidates the validation.

### Using LLM-as-a-Judge to classfify and evaluate

The naive approach is to curate questions, run the RAG, hand-label every response and derive the metrics from there. At 60 questions times 3 configs equals 180 responses, that fits in an afternoon. But it breaks the moment the benchmark grows. Imagine 600 questions, or 6,000, or the team wants to re-run the eval every time the underlying LLM is updated. Hand-labelling is a bottleneck that does not scale.

The scalable answer is to have an LLM act as the judge. But that is only defensible if you can prove the judge agrees with a human labeller. That proof is Cohen's kappa.

Cohen's kappa is a number between 0 and 1 that measures how well two labellers agree, after correcting for the agreement you would expect by chance. By convention, kappa above 0.6 is "substantial agreement" and kappa above 0.8 is "almost perfect." If the LLM judge has a high enough kappa against a small hand-labelled validation set, you can trust it on the rest. The benefit is that the hand-labelling work is a one-time cost. After that, expanding the benchmark or re-running it on a new model is just a single command.

**How we validated the judge.**

1. Run the LLM judge over all 180 responses. The judge classifies each into one of 5 labels.

   **On answerable questions:**
   - `CORRECT_ANSWER`. The model produced a substantively correct answer that matches the ground truth, or a defensible sibling code from the retrieved context.
   - `WRONG_ANSWER`. The model committed to a specific answer, but the answer is wrong and not defensible from the retrieved context.
   - `OVER_REFUSAL`. The model refused or said "I don't know" on a question it should have been able to answer. The opposite failure mode from hallucination.

   **On unanswerable questions:**
   - `CORRECT_ABSTAIN`. The model correctly recognised it cannot answer and declined to make a factual claim.
   - `HALLUCINATION`. The model committed to a specific factual claim (a code, an attribute, a year, a mapping) that the corpus does not support. Hedged claims like "this might be SSOC 25121" count.

   The judge also returns a `retrieval_provided_answer` boolean for each response. This separates "retrieval failed" hallucinations (the right chunk was never surfaced) from "model fabricated despite good retrieval" (the right chunk was in the context but the model invented anyway).

2. Sample 60 of the 180 responses into a 30-question dev set and a 30-question test set, stratified across categories with a fixed random seed (42).
3. Hand-label all 60 questions.
4. Iterate the judge prompt on the dev set if needed until kappa is high.
5. Measure final kappa on the held-out test set. The test set is reported only once. This prevents tuning to the test set.

**Result.** Dev kappa = 1.000 and test kappa = 1.000. No prompt iteration was needed because the rubric agreed with the hand labels on every sampled cell on the first try. The prompt is now locked in `src/eval/judge_prompt.py` at version `v0`. Any future edit invalidates the kappa and would require re-measurement.

---

## Results

The full charts (bar plots, per-category heatmaps, failure-mode examples) live in `notebooks/results_analysis.ipynb`. The key numbers are reproduced below.

### Headline 3-way comparison

Both tables use the same formula in `src/eval/metrics.py`: `(count of one judge label) / (count of rows in the relevant subset)`, per config. For the headline 3-way comparison, the denominator is 40 unanswerable rows for hallucination_rate and correct_abstention_rate, and 20 answerable rows for the three answerable-side rates. For the per-category breakdown, the denominator is filtered to one category (10 questions per unanswerable category per config). So strict's 0.075 headline hallucination rate is 3 out of 40, and its 0.20 SSIC-confusion rate is 2 out of 10.

20 answerable plus 40 unanswerable per config (n=60 per column).

| Metric | neutral | forced | strict |
|---|---:|---:|---:|
| Hallucination rate | 0.150 | 0.825 | **0.075** |
| Correct abstention rate | 0.850 | 0.175 | **0.925** |
| Over-refusal rate | 0.000 | 0.000 | 0.000 |
| Correct answer rate | 1.000 | 1.000 | 1.000 |
| Wrong answer rate | 0.000 | 0.000 | 0.000 |

### Per-category hallucination rate

| Category | neutral | forced | strict |
|---|---:|---:|---:|
| Beyond-corpus attribute | 0.00 | 0.90 | 0.00 |
| Obsolete version | 0.10 | 1.00 | 0.00 |
| False premise | 0.10 | 0.50 | 0.10 |
| SSIC confusion | **0.40** | 0.90 | **0.20** |

### Overall Findings

- Strict hallucinates on 7.5% of unanswerable questions while neutral hallucinates on 15%. The gap is the size of the effect the prompt is actually buying. At the same time, all three configs answer 100% of the answerable questions correctly with 0% over-refusal anywhere. That rules out the "score perfectly on hallucination by refusing everything" trap, which is the failure mode a hallucination-only metric would miss.

- Forced hallucinates on 82.5% of unanswerable questions, neutral on 15%, and strict on 7.5%. That 11x range, driven only by prompt wording, tells us three things at once. First, the judge is detecting hallucinations correctly: if forced had scored low, the judge would be mislabelling and every other number in the table would be suspect. Second, the prompt is genuinely the active lever, because the model can be moved across that range just by changing the instruction text. Third, there is no silent server-side guardrail filtering hallucinated outputs on the provider's end, because if there were, forced would not have been able to push the rate this high. The headline finding ("strict cuts hallucination roughly in half") is therefore a real prompt effect, not noise or an artefact of upstream model behaviour. Prompt engineering is the right level of intervention to invest in.

- **SSIC confusion is the residual weak category, even under strict.** 20% of SSIC questions still produce an SSOC code, which is a hallucination. This particular failure mode allows a team to know what to fix next (topic gating, a clarifying-question step, or a classifier in front of retrieval, or tool calling indexes). 

- All hallucinations are generation-side, not retrieval-side.Recall@4 is 100% on answerable questions, and the `retrieval_provided_answer` flag shows the right chunks were available in nearly every hallucination cell (31 of 33 forced, 6 of 6 neutral, 3 of 3 strict). The model fabricated regardless of having the right chunks. This kind of attribution is exactly what the harness was built to produce: it tells the team the next lever is the prompt or the generator, not the retriever.

---

## How to Run

**Prerequisites.** Docker and Docker Compose. A Google AI Studio API key with Gemini access is required for the first-time vector-index build, and for live queries or eval reproduction. Browsing the pre-computed eval results in the notebook does not require an API key.

The supported path is Docker. The image bundles the source code, the SSOC 2024 source files, the JSONL chunks, and the pre-computed eval outputs. The Chroma vector index is built locally on first run, because its binary format is not portable across machines.

```bash
# 1. Clone
git clone <repo-url>
cd KnowOrNoBench

# 2. Configure the API key
cp .env.example .env
# Open .env and paste your GEMINI_API_KEY. This is needed for the first-time
# index build below, and for any live queries. If you only want to browse the
# pre-computed results in the notebook, you can skip this step and the
# build-index step, but the live query commands will not work.

# 3. Build the image (about 5 minutes the first time)
docker compose build

# 4. (First time only) Build the Chroma vector index inside the container.
#    About 5 minutes, ~$0.02 in embedding spend. The index is written to
#    ./chroma_db on your host via a volume mount, so this step is one-time only.
docker compose run --rm rag python -m scripts.build_index

# 5. Launch the environment
docker compose up

# 6. Open http://localhost:8888 in your browser.
#    Open notebooks/results_analysis.ipynb to see the headline 3-way comparison,
#    per-category breakdowns, and failure-mode examples.
#    No API key needed for this path.
```
### Running commands while `docker compose up` is in the foreground

We can run additional commands on a second terminal window using `docker exec` to attach to the already-running container by name (`knowornobench`). Open another terminal window and run the following:

```bash
# Live query against the RAG (requires GEMINI_API_KEY in .env)
docker exec -it knowornobench python -m scripts.query "What is SSOC 25121?" --config strict

# Try a question the corpus cannot answer to see strict refuse cleanly
docker exec -it knowornobench python -m scripts.query "What is the SSOC code for a unicorn trainer?" --config strict

# Same question under forced and neutral to see the three-way contrast
docker exec -it knowornobench python -m scripts.query "What is the SSOC code for a unicorn trainer?" --config forced
docker exec -it knowornobench python -m scripts.query "What is the SSOC code for a unicorn trainer?" --config neutral

# Reproduce the full evaluation (~15 min for RAG, ~25 min for judge, ~$0.90 total)
docker exec -it knowornobench python -m scripts.run_full_eval --rebuild
docker exec -it knowornobench python -m scripts.run_judge     --rebuild

```

**If a run stops halfway or errors out on some cells.** Both `run_full_eval` and `run_judge` persist their output to disk after every cell, so a Ctrl-C, a network blip, or a rate-limit error never loses more than one in-flight call. Both scripts also resume by default: if you re-run them WITHOUT `--rebuild`, every cell that already has a valid response or label is skipped, and only the missing or errored cells are retried. Cells that errored mid-run leave the response column blank and the error message in a separate `*_error` column, which is exactly the condition the resume logic looks for. So to recover, just drop the `--rebuild` flag and run the same command again:

```bash
# Resume after a stop or error (no --rebuild flag, so already-done cells are skipped)
docker exec -it knowornobench python -m scripts.run_full_eval
docker exec -it knowornobench python -m scripts.run_judge

# Once everything is done, re-running again prints "skipped: 180, processed: 0" and exits.
# That is how you confirm there is nothing left to do.
```

If you want to inspect what actually failed before retrying, open `results/responses.xlsx` (for RAG errors) or `results/judged_{config}.xlsx` (for judge errors) and look at the `*_error` columns. They contain the raw exception text, which is usually enough to tell whether the issue was a quota limit (wait and retry), a network blip (just retry), or a real bug (open an issue).

**After re-running the evaluation:** the notebook reads the Excel result files into memory at cell-run time, so it keeps showing the old numbers until you re-execute the cells. After `run_full_eval` and `run_judge` finish, refresh the notebook in your browser and choose `Kernel → Restart and Run All` to render the fresh tables and charts.

### Stopping the environment

```bash
docker compose down            # stop the container
```

---

## Limitations

- **Same-family judge.** Originally the intended judge model was `llama-3.3-70b-versatile` via Groq and `DeepSeek V4 Flash (free)` via OpenRouter, chosen so the judge came from a different model family than the generator (`gemini-2.5-flash`). This guards against self-preference bias, where a judge might rate its own family's outputs more favourably. However, mid-project, the free tier for both models got rate limited and subsequently switched to `gemini-2.5-pro` as I still had some credit for it left.
- **N=30 per dev/test split.** This gives a wide confidence interval on kappa. The point estimate of 1.000 should be read as a range. A production-scale validation would use 90 or more labels per split and allows kappa to be more reliable, and concluding that the LLM used for judging can be trusted.
- **Per-category n=10.** Confidence intervals on per-category rates are wide. We do not run statistical significance tests across configs at this sample size.

**What I would do without time or budget limits.**

The current submission has a working evaluation harness, but several gaps in the methodology are honest limitations rather than mistakes. With more time or budget, here is what would close each one.

- **A paid cross-family judge.** The current judge is `gemini-2.5-pro`, which is the same model family as the `gemini-2.5-flash` generator. Same-family judges carry a risk of self-preference bias, where the judge rates outputs from its own family more favourably than an outside model would. With budget, I would re-measure κ using a non-Gemini judge such as paid Anthropic Claude, OpenAI GPT-4, or paid OpenRouter Llama 3.3 70B. If the cross-family κ matches the current value, the methodology is validated. If it diverges, the cross-family κ is the one I would publish, and the gap between the two would itself be a finding worth reporting.

- **A larger benchmark (N=300 instead of N=60).** The current 60-question benchmark might not be enough to truly evaluate the system. That is why the strict-vs-neutral gap (7.5% vs 15%) is described as a directional finding rather than statistically significant. Scaling the benchmark to 300 questions (roughly 60 per category instead of 10 to 20) would tighten the confidence interval, which is the threshold where the strict-vs-neutral comparison becomes a defensible significance claim. The per-category breakdown would also become much more reliable, especially for SSIC confusion where the residual 20% hallucination rate currently sits on only 10 questions.

- **A `PARTIAL_ANSWER` label and inter-rater agreement on the questions themselves.** The current 5-label rubric collapses every non-committal response into either `CORRECT_ABSTAIN` or `HALLUCINATION`, but in practice we sometimes encounter middle-ground responses where the model acknowledges related context without committing to a specific claim. Adding a sixth label, `PARTIAL_ANSWER`, would let the eval distinguish "the model is hedging, which is closer to honest abstention" from "the model is committing to a false specific claim, which is a clean hallucination," and give a more honest signal on borderline cells. The same gray-zone exists at curation time. If two independent labellers both think a question I designed as unanswerable can actually be partially answered from the corpus, that is a strong signal the question is miscategorized and would corrupt the per-category breakdown if used as-is. With time, I would run inter-rater agreement on the questions themselves, not just on the judge labels, so questions where labellers disagree on the expected behaviour get rewritten or dropped before they enter the frozen benchmark.

- **A user-acceptance-test pass on the benchmark before it is frozen.** The current benchmark is the work of a single curator (me) drafting questions against the corpus and verifying each one against the chunks. A more defensible flow is to run a UAT where two or three independent reviewers attempt to answer each question using only the corpus, and report whether they could answer it, what code or claim they would have given, and whether the question felt clear. Questions that produce inconsistent reviewer behaviour get adapted or replaced. The output is a curated set with documented agreement on which questions are answerable, which are not, and why. That turns the benchmark itself into a measured artifact rather than a hand-drafted one, which is the same scaling story as LLM-as-judge: spend the human effort once on validation, then trust the artifact for repeated measurement.

- **Blind double-labelling for inter-rater κ on the hand labels themselves.** Right now I am the only person who hand-labelled the 60-cell κ validation set. The judge-vs-me κ is what is reported, but there is no measurement of the rubric's reliability when applied by a second independent human. With time, I would have a second labeller go through the same 60 cells without seeing my labels and compute the human-vs-human κ. A high inter-rater κ would confirm that the rubric is objective and that the judge-vs-me agreement is meaningful. A low one would mean the rubric itself is ambiguous and needs tightening before any judge can be validated against it. This is a real methodological hole in the current submission.

- **Multiple samples per cell at non-zero temperature.** The eval runs each cell once at temperature 0.0, which makes the run deterministic but produces a single point estimate per cell with no variance bars. Production deployments often use a non-zero temperature for output diversity, and a deterministic temperature-0 measurement can overstate reliability. With budget, I would re-run each cell five times at temperature 0.7 and report not just the hallucination rate but the standard deviation across samples. That would give a deploying team a realistic sense of how much variation in behaviour to expect when their actual system is not pinned to temperature 0.

- **Ablating beyond just the prompt.** The current eval only varies one component of the RAG, the prompt, while holding everything else fixed (same retriever, same embedding model, same generator, same top-k, same chunking). The conclusion that "prompt engineering is the right lever" is therefore only defensible for this corpus and this model combination. With more time, I would run the same benchmark against ablated versions of the rest of the pipeline: a swapped embedding model (such as `text-embedding-3-large` or BGE) to test retrieval robustness, a swapped generator (such as GPT-4o or Claude) to see whether the strict-abstention behaviour transfers across model families, different top-k values (1, 4, 8) to measure how much context the model actually uses, and a cross-encoder reranker after dense retrieval to test whether better ranking moves the residual SSIC confusion rate.

- **Swap Chroma for an enterprise-grade vector store in production.** Chroma is a great choice for this submission: it is local, persistent, free, and supports the explicit-cosine setup the project needs. It is not the right choice for production at scale. For a deployed RAG, I would move the index to a managed vector store such as Azure or GCP, depending on the deployment environment. The advantages include things like horizontal scaling beyond a single machine, high availability and managed backups, role-based access control and per-tenant isolation, hybrid retrieval (vector plus keyword) for harder query types and native monitoring. 


---

## Deployment considerations

**Who would run it and where.** Any team running a RAG over a knowledge base. The eval harness is offline. It reads frozen artifacts (the benchmark file, the responses Excel, the judged Excel) and produces a report. It would naturally live as a scheduled job, run before each release or each time the underlying LLM is upgraded. The RAG-under-test is a separate concern. In this project it is a local Chroma index plus Gemini API calls.

**Cost at scale.** The per-query RAG cost is small. Embedding plus generation is roughly $0.0003 per query on Gemini Flash, which is about $1 per day for 10,000 internal queries. The judge costs more because it uses Pro: a full re-judge of the 180-response benchmark is about $0.85. Scaling the benchmark to 600 questions would cost about $2.50 per re-judge. The kappa-validation hand-labelling effort is the only step that does not scale with budget alone (it requires human time once).

**What I would monitor in production.** Three signals. (1) Hallucination rate on a fixed held-out set; if it drifts upward after a model upgrade, the model is regressing. (2) Recall@4 on the answerable subset; if retrieval recall drops, the index or the embedder has degraded. (3) Abstention rate distribution; a sudden drop means the model has become over-confident.

**The one specific risk that would keep me up at night.** The abstention behaviour depends entirely on a single prompt instruction. The model could silently start ignoring that instruction after a Gemini model upgrade, and there is no decode-time enforcement to catch it. A model refresh could move the strict-config hallucination rate from 7.5% to 30% without any code change on our side. The mitigation is to re-run the full eval on every Gemini model version bump and to gate releases on hallucination rate, so a regression cannot ship silently.

---

## Free-tier note

The RAG itself runs end-to-end on free-tier resources. Embeddings come from `gemini-embedding-001` on the free tier (1,000 requests per day), generation runs on `gemini-2.5-flash` on the free tier, and Chroma is a local vector database with no hosting cost.

The judge step in this submission used `gemini-2.5-pro`, which is paid. Total project spend on the judge was about $0.85, after the originally-intended free-tier `Llama 3.3 70B` via Groq and `DeepSeek V4 Flash` via OpenRouter exhausted mid-run. 

What scales with budget: a larger benchmark, a cross-family judge for stronger kappa guarantees, and multiple samples per cell at non-zero temperatures to see variance.

---

## License

MIT.
