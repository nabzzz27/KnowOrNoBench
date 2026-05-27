# Process

A narrative of how this project was built. What was tried, what was dropped, which tools made which decisions, and where I exercised judgment.

## Picking the problem

The brief listed roughly fifteen example problem sketches. The one that fit best was "out-of-knowledge-base robustness" for a RAG. The brief weights "Evaluation and effectiveness" at 30%, which is the largest single dimension, so investing in a rigorous evaluation methodology rather than an impressive RAG was the right way to spend the time.

The dataset I chose is the Singapore Standard Occupational Classification 2024 (SSOC 2024), published by SingStat. The public-sector framing fits the brief, the corpus is small and narrow (which is the right shape for "knows when it does not know" experiments), and the data has clean provenance. I did consider other Singapore-based sources, but several were either hard to scrape reliably or had terms of use that did not permit redistribution.

## A short de-risking spike before any production code

Before writing the real pipeline, I wrote one throwaway script that ran six small probes. Each probe was designed to retire one risky assumption: does the stack install on my Python version, does the source data parse without cleanup, does cosine retrieval actually surface the right chunks, does the embedding API behave the way the documentation claims. The script also had a dry-run mode that printed exactly what would be sent to the embedder, with zero API calls, so the data going in was inspectable before any money was spent.

The spike paid for itself in two ways. It confirmed the architecture choices in advance (structure-aware chunking, asymmetric embeddings, Chroma with cosine similarity, a Flash generator) so I could build the real pipeline with confidence. It also surfaced a rate-limit characteristic I had not expected: the embedding API counts quota per text and per minute, not per request, which meant my first naive retry approach was useless. I replaced it with a proactive pacer that throttles requests up front, and that became the production behaviour.

## Building the index, and hitting quota walls on both sides

The real pipeline lifted the spike's pacer and retry logic into proper modules. Embedding lives in `src/embed.py` with two exported functions, `embed_documents` and `embed_query`, named to enforce the asymmetric task-type pattern at the call site. The vector store wraps Chroma with cosine similarity set explicitly, so the metric travels with the index.

The first surprise was that the binding constraint on the free tier was not the per-minute rate the spike had measured. It was a separate daily cap that the spike never exposed, because the spike only embedded a small sample. The full corpus build hit the daily cap partway through and could not recover until the next day. The thing that saved the run was that I had built the index step to be idempotent and crash-resumable from the start. It queries existing chunk IDs first and only embeds the missing ones, so already-indexed chunks survived across sessions.

The second quota wall came on the generation side. The free tier for the generator had a small daily request cap, and running the full eval across multiple prompts would have taken many days at that rate. I made the deliberate decision to enable billing for both embedding and generation. The brief explicitly permits this, the total project cost was forecast to be well under one dollar, and the alternative was a multi-day calendar slip on the eval. Documenting the decision as a deliberate trade-off rather than hiding it was important.

A third surprise came much later, during the dockerization step. The Chroma vector index built on my host could not be loaded inside the Docker container; the HNSW segment file is a platform-dependent binary blob and does not survive a move between Python builds. The fix was to stop shipping the index in the repo and document a one-time `python -m scripts.build_index` step that a reviewer runs inside the container after the image is built. The chunks JSONL is portable and is shipped, so the rebuild only needs the embedding step and finishes in a few minutes.

## Verifying retrieval quality before building generation

Before writing any answer-generation code, I wanted a defensible answer to one question: are the embeddings good enough that the right chunk is reachable in the first place? Generation-side problems can be debugged later by tweaking prompts. Retrieval-side problems cannot, because if the right chunk never makes it into the context, no amount of prompt work can save the response.

I separated the two failure modes with a small notebook that runs hand-curated queries against the live index and captures the rank, the distance, and the full chunk text for each retrieval. The headline finding was that recall at the top four retrieved chunks was perfect on the answerable queries, which meant retrieval would not be the bottleneck. The recall-at-one misses were all parent-versus-child or sibling confusions, inherent to having both four-digit and five-digit codes in the same index. They are harmless with top-four retrieval because the gold chunk is always in the context window. This let me move on to the prompt work with confidence.

## Adding the forced prompt configuration

The first version of the RAG had two prompt configurations: a naive one (just use the context) and a strict one (use the context, refuse if it does not support an answer). I expected the naive prompt to hallucinate on unanswerable questions and the strict prompt to refuse, and the gap between them would be the headline finding.

The first smoke test broke that expectation. On both test unanswerable questions, both prompts refused. The generator was honest-by-default at temperature zero, so the two-prompt contrast collapsed. There was no explicit counterfactual that removed the model's natural caution, which meant the eval could only measure "does the abstention rule polish behaviour the model already has," rather than "does the rule prevent hallucination the model would otherwise produce."

The fix was a third configuration, `forced`, which explicitly forbids refusal language and instructs the model to commit to an answer no matter what. Nobody would ship this prompt in production. It exists only inside the eval, as a deliberate contrast to `strict` and as a positive control for the judge. By construction, every `forced` response on an unanswerable question is a hallucination. If the judge ever labels one as a correct refusal, the judge is broken. This is what gives the eval ground truth on a large slice of cells without hand-labelling, and it is what lets the judge κ be validated meaningfully.

## Curating the 60-question benchmark

The naive approach to benchmark curation would be to write a handful of questions, run the system on them, and inspect the outputs by eye. That works for a smoke test, but it does not produce a defensible measurement, because there is no way to control what the questions actually test. I wanted each question to provoke a specific failure mode, so that the per-category breakdown would surface actionable signal rather than just an aggregate hallucination rate.

I settled on sixty questions, hand-curated. Sixty is small enough to draft and verify against the corpus in the available time, and large enough to give meaningful per-category rates (about ten questions per failure-mode category). Each question was drafted against the actual corpus, with the answerable codes and their chunk excerpts verified verbatim. The benchmark was frozen in git before any RAG output was generated against it, which is the defence against picking questions after seeing the answers.

The category structure adapts RAGTruth (Niu et al., ACL 2024), a peer-reviewed RAG hallucination corpus that catalogued the four recurring ways RAG systems fail: evidence-conflicting, subjective or unsupported, out-of-context, and baseless information. The four unanswerable categories in this benchmark each map onto one of those failure types, re-grounded in the SSOC domain (for example, "SSIC confusion" is the Singapore-specific instance of out-of-context, where the user asks about the adjacent industry classification). A fifth answerable category was added as a control, because an eval that only measures hallucination could be gamed by a system that refuses everything. Anchoring to a peer-reviewed taxonomy made the category structure defensible rather than arbitrary.

## The LLM-as-judge, its categories, and the retrieval-attribution flag

Hand-labelling every response works once, but it does not scale. The moment the benchmark grows or the team wants to re-run after a model upgrade, hand-labelling becomes a bottleneck. The scalable answer is to have an LLM act as the judge, but a judge is only defensible if you can prove it agrees with a human labeller. Cohen's κ against a small hand-labelled validation set is the standard discipline, and it is what gates whether the judge's labels can be trusted on the rest of the data.

The judge classifies each response into one of five labels. On answerable questions: `CORRECT_ANSWER` for a right answer (including a defensible sibling code drawn from the retrieved context), `WRONG_ANSWER` for a specific incorrect claim, and `OVER_REFUSAL` for refusing a question that had a defensible answer. On unanswerable questions: `CORRECT_ABSTAIN` for correctly declining to commit, and `HALLUCINATION` for making a specific claim the corpus does not support. The five labels were chosen to capture both failure directions in a single rubric, so that a system that scores well by refusing everything is visible as over-refusal rather than as a perfect hallucination score.

The judge also returns a `retrieval_provided_answer` boolean. This separates two failure modes that look identical on the label alone: cases where the right chunk was never retrieved, and cases where the right chunk was in the context but the model fabricated anyway. The attribution matters because the two failure modes have different fixes. Retrieval-side failures are addressed by a better embedder or retriever. Generation-side failures are addressed at the prompt or the model. Without this split, the eval would tell the team that something is wrong, but not which lever to pull.

## What surprised me

The generator's honest-by-default behaviour collapsed the original two-prompt design on the first smoke test, which forced the redesign into three prompts. The daily quota cap on the embedding API was the binding constraint, not the per-minute window the spike had measured. Retrieval recall at the top four chunks held up across the whole project, so the retrieval-improvement work I had budgeted for was not needed and the action all sat on the generation side.

## What I would do differently

Build the evaluation methodology before the RAG, not after. If I had drafted the eval first, the two-prompt methodology gap would have been caught at design time rather than at the first smoke test. Budget for paid judge cost upfront, rather than discovering it mid-run when a free tier exhausts. Lock the reproducibility contract earlier in the project by drafting the Dockerfile during the first week rather than at the end; the version that ships now bundles the source files, the pre-built index, and the eval outputs so a reviewer can see the harness running from a clean clone, but having that contract from day one would have made every intermediate decision (where artifacts live, what gets gitignored, how secrets are passed) easier.
