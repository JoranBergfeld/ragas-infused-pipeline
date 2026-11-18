# Eval history

The committed `eval_results_<mode>.json` files only ever hold the latest run. This is the
log of what they said before, so a re-run doesn't quietly erase the last baseline.

Newest first. Means over `data/testset.jsonl`. With n in the tens, treat anything inside
the confidence intervals as "no measurable difference" (ADR-0018).

## 2026-11-18: all nine modes on EmbeddingGemma 2 (#47, #50)

This is the first full run after homelab's `rag` tables were re-embedded with `google/embeddinggemma-2` on the Spark:
- 768 dims;
- query prompt `task: question answering | query: `;
- document prompt `title: none | text: `.

**Setup.** The corpus is the same 2026-10-24 export. These are unchanged from the run before:
- the 43 questions;
- the Spark reranker (`nvidia/llama-nemotron-rerank-1b-v2`);
- the Foundry generator;
- the online judge;
- the offline judge (`DeepSeek-V4-Pro`), on ragas 0.4.3 and Python 3.11.

It ran from the laptop through an SSH tunnel to the Spark, on code identical to ac5a418 (#50 merged). Retrieval used the read-only `rag_service` login. The "before" numbers are the text-embedding-3-small@1536 files from bd3ec63.

**Two things besides the model changed:**

- **`raptor_sac` leaves embed differently.** Since #46 they embed `context + content`, where before they embedded the content alone. Its delta is model plus input.
- **The first attempt was thrown away.** Two bugs made it invalid, both fixed in #50:
  - The agentic planner's chat calls went to the embedding server and 404'd. All four `*_agentic` modes silently ran single-query (#49).
  - On Windows, the testset was read as cp1252, so the 7 questions with an em-dash went out as mojibake.

  These numbers are from the clean rerun. The agentic records show 2–3 sub-query rounds per question.

te3s → Gemma. `*` marks a te3s mean outside the new run's 95% interval:

| mode | hit_rate@fused | mrr@reranked | faithfulness | answer_relevancy | context_precision | context_recall | abstained |
|---|---|---|---|---|---|---|---|
| contextual | 0.911 → 0.937 | 0.779 → 0.790 | 0.853 → 0.867 | 0.846 → 0.885 | 0.709 → 0.685 | 0.791 → 0.838 | 0.093 → 0.047 |
| baseline | 0.966 → 0.903 | 0.811 → 0.759 | 0.896 → 0.879 | 0.878 → 0.873 | 0.716 → 0.677 | 0.869 → 0.798 | 0.093 → 0.023* |
| raptor_sac | 0.966 → 0.928 | 0.813 → 0.790 | 0.876 → 0.898 | 0.842 → 0.873 | 0.729 → 0.681 | 0.820 → 0.808 | 0.023 → 0.116 |
| graphrag | 0.956 → 0.902 | 0.571 → 0.568 | 0.818 → 0.798 | 0.743 → 0.813 | 0.463 → 0.416 | 0.321 → 0.352 | 0.279 → 0.349 |
| combined | 0.984 → 0.959 | 0.760 → 0.746 | 0.892 → 0.905 | 0.895 → 0.873 | 0.603 → 0.577 | 0.672 → 0.607 | 0.140 → 0.093 |
| baseline_agentic | 0.972 → 0.980 | 0.818 → 0.803 | 0.850 → 0.882 | 0.864 → 0.827 | 0.538 → 0.527 | 0.717 → 0.731 | 0.093 → 0.047 |
| raptor_sac_agentic | 0.967 → 0.974 | 0.857 → 0.810 | 0.878 → 0.901 | 0.865 → 0.876 | 0.552 → 0.524 | 0.759 → 0.768 | 0.047 → 0.116 |
| graphrag_agentic | 0.974 → 0.984 | 0.600 → 0.610 | 0.804 → 0.817 | 0.821 → 0.846 | 0.453 → 0.424 | 0.381 → 0.297 | 0.302 → 0.395 |
| combined_agentic | 0.988 → 0.994 | 0.786 → 0.777 | 0.888 → 0.871 | 0.877 → 0.852 | 0.572 → 0.565 | 0.654 → 0.683 | 0.047 → 0.023 |

Paired per question (all 43 pair up), no metric in any mode passes the randomization test at p < 0.05. Overall the switch is a wash: the multimodal option costs nothing measurable on text.

Where the bootstrap interval on the paired difference excludes zero, though the randomization test doesn't confirm it:

- **`baseline`:**
  - hit_rate@fused −0.062 [−0.140, −0.008], p = 0.13.
  - mrr@reranked −0.052 [−0.109, −0.009], p = 0.06.
  - context_recall −0.067 [−0.147, −0.004], p = 0.13.

  The hit-rate drop is exactly what the retrieval-only check predicted. Without the situating context, Gemma's dense leg finds fewer gold pages. This is the one mode where the switch probably costs something.
- **`raptor_sac` abstained:** +0.093 [+0.023, +0.186], 4 more abstentions, p = 0.13. The model and the leaf input both changed, so this doesn't separate them.
- **`combined_agentic` answer_relevancy:** −0.045 [−0.105, −0.001], p = 0.08.
- **Every agentic mode:** hit_rate@fused is up +0.006 to +0.009, with intervals touching zero. One question gained per mode at most.

**Coverage.**
- faithfulness and answer_relevancy are scored on 26–42 of 43. They're lowest for `graphrag` (28) and `graphrag_agentic` (26), because abstentions aren't scored.
- context_precision is scored on 41–43, context_recall on 42–43.
- Lost judge jobs: five DeepSeek 429s and two timeouts. The reranker never reported `reranker-offline`.

## 2026-11-16: EmbeddingGemma 2 settings, retrieval only (#47)

This isn't a mode run, and no `eval_results_*.json` changed. It was a check, before
switching the embedder to `google/embeddinggemma-2` on the Spark (chosen for
multimodality, homelab #220), of which Gemma settings to use and how big the regression
is against `text-embedding-3-small@1536`. Run with `scripts/compare_embeddings.py`. The
2026-10-24 export's `ms_docs` and `baseline` (7,944 rows each) were loaded into scratch
databases, and the 43 hand-authored questions went through the production
`HybridSubstrate` (dense 15 + BM25 15, RRF k=60). Scoring is URL hit rate / MRR over the
whole stage, as the eval does it. There's no reranker, generator or judge. The Gemma
prompts are `task: search result | query: ` (search) or `task: question answering | query: `
(qa) on queries, and `title: none | text: ` (none) or `title: {title} | text: ` (title) on
documents.

`ms_docs` (the contextual table):

| variant | hit@dense | mrr@dense | hit@fused | mrr@fused | mrr@fused top 5 |
|---|---|---|---|---|---|
| te3s 1536 | 0.871 | 0.793 | 0.919 | 0.810 | 0.800 |
| g768 search none | 0.874 | 0.760 | 0.933 | 0.763 | 0.755 |
| g768 search title | 0.846 | 0.755 | 0.928 | 0.779 | 0.770 |
| **g768 qa none** | 0.878 | 0.757 | 0.937 | 0.788 | 0.780 |
| g768 qa title | 0.850 | 0.751 | 0.933 | 0.765 | 0.757 |
| g512 search none | 0.866 | 0.740 | 0.933 | 0.758 | 0.751 |
| g512 search title | 0.846 | 0.746 | 0.928 | 0.732 | 0.721 |
| g512 qa none | 0.874 | 0.743 | 0.933 | 0.775 | 0.768 |
| g512 qa title | 0.846 | 0.753 | 0.928 | 0.783 | 0.772 |

`baseline` (plain chunks, no situating context):

| variant | hit@dense | mrr@dense | hit@fused | mrr@fused | mrr@fused top 5 |
|---|---|---|---|---|---|
| te3s 1536 | 0.938 | 0.760 | 0.966 | 0.779 | 0.769 |
| g768 search none | 0.855 | 0.695 | 0.899 | 0.754 | 0.742 |
| g768 search title | 0.865* | 0.711 | 0.922 | 0.779 | 0.768 |
| **g768 qa none** | 0.876 | 0.713 | 0.903 | 0.803 | 0.799 |
| g768 qa title | 0.871 | 0.699 | 0.927 | 0.779 | 0.766 |
| g512 search none | 0.863 | 0.716 | 0.899 | 0.789 | 0.782 |
| g512 search title | 0.857* | 0.712 | 0.922 | 0.755 | 0.745 |
| g512 qa none | 0.845 | 0.690 | 0.903 | 0.777 | 0.769 |
| g512 qa title | 0.876 | 0.713 | 0.927 | 0.782 | 0.767 |

`*` = paired randomization test vs te3s, p < 0.05.

Recommended: **768 dims, `question answering` query prompt, `title: none`**. The
differences are mostly inside the noise, but they point one way:

- **768 over 512:** 768 is ahead in nearly every cell, though never measurably. At 8k
  rows per table the storage saving is irrelevant.
- **qa over search:** fused MRR is higher in both tables (0.788 vs 0.763, 0.803 vs 0.754).
  Caveat: every test item is a question, so this may flatter `qa` on question-shaped traffic.
- **No title:** titles don't help, and on `ms_docs` they cost ~0.03 dense hit. The
  situating context already starts with a title breadcrumb, so the title is repeated.

How big is the regression, g768 qa none vs te3s (paired differences, 95% bootstrap):

- `ms_docs`: no measurable difference. hit@fused +0.018 [-0.025, +0.079],
  mrr@fused -0.022 [-0.108, +0.064], mrr@dense -0.036 [-0.137, +0.062]. 3 questions
  gained, 2 lost.
- `baseline`: **hit@fused -0.062 [-0.140, -0.008]**, which is 4 questions lost and none
  gained. The bootstrap interval excludes zero, though the randomization test doesn't
  (p = 0.13). mrr@fused +0.025 [-0.047, +0.098]. Without the situating context, Gemma's
  dense leg finds fewer gold pages than text-embedding-3-small. The rerank only sees
  `fused`, so the `baseline` mode will probably lose a little after the switch.

The next full eval after `ingest --replace` is what counts. This only checks the first
stage.

## 2026-10-25: the rerank modes on the Spark reranker (#40)

`contextual`, `baseline` and `raptor_sac` are back, reranked by
`nvidia/llama-nemotron-rerank-1b-v2` on the DGX Spark instead of Azure's semantic ranker
(ADR-0021 amendment). Same 43 questions, generator, judge (`DeepSeek-V4-Pro`), ragas 0.4.3
and Python 3.11 as the run below. Ran from the laptop through an SSH tunnel to the Spark.

Not like for like with the "before" numbers, the Azure-era files from f87ec34 (2026-09-13):
both the reranker and the first stage changed (pgvector + Postgres BM25 instead of Azure
Search). So this says how the whole swap landed, not which reranker is better.

Azure → Spark. `*` marks an Azure-era mean outside the new run's 95% interval:

| mode | hit_rate@fused | hit_rate@reranked | mrr@reranked | faithfulness | answer_relevancy | context_precision | context_recall | abstained |
|---|---|---|---|---|---|---|---|---|
| baseline | 0.966 → 0.966 | 0.875 → 0.910 | 0.829 → 0.811 | 0.854 → 0.896* | 0.871 → 0.878 | 0.688 → 0.716 | 0.754 → 0.869* | 0.023 → 0.093 |
| contextual | 0.916 → 0.911 | 0.852 → 0.851 | 0.815 → 0.779 | 0.902 → 0.853* | 0.908 → 0.846 | 0.670 → 0.709 | 0.757 → 0.791 | 0.093 → 0.093 |
| raptor_sac | 0.911 → 0.966* | 0.855 → 0.897 | 0.797 → 0.813 | 0.899 → 0.876 | 0.874 → 0.842 | 0.721 → 0.729 | 0.773 → 0.820 | 0.070 → 0.023 |

95% bootstrap intervals on the new run:

| mode | faithfulness | answer_relevancy | context_precision | context_recall | abstained |
|---|---|---|---|---|---|
| baseline | 0.86–0.93 | 0.83–0.92 | 0.60–0.82 | 0.77–0.95 | 0.02–0.19 |
| contextual | 0.81–0.89 | 0.77–0.91 | 0.59–0.81 | 0.68–0.89 | 0.02–0.19 |
| raptor_sac | 0.81–0.93 | 0.77–0.90 | 0.62–0.83 | 0.71–0.91 | 0.00–0.07 |

What it says, with the usual 43-question caveat:

- Overall it's a wash. One mode got better, one got a bit worse, one didn't move.
- `baseline` improved. Faithfulness and context_recall both left their old values outside the
  interval. It abstains more (2% → 9%), but that's inside the noise.
- `contextual` lost faithfulness (0.902 → 0.853, outside the interval) while hit rate held and
  both context metrics went up. So it gets the same documents and writes answers the judge
  likes less. No idea why yet. Answer relevancy dropped too, inside the noise.
- `raptor_sac`: no judge metric moved beyond noise. Its fused hit rate went up (0.91 → 0.97),
  which is the first stage, not the reranker.
- The reranker itself never failed a call: zero timeouts, zero `reranker-offline`.

Coverage: faithfulness and answer_relevancy scored on 39/43 for `baseline` and `contextual`,
42/43 for `raptor_sac`; context_precision 42/43 everywhere. Lost judge jobs: three 429s
from the DeepSeek deployment and one Azure `content_filter` 400 on a guardrail call.

## 2026-10-25: retrieval on pgvector (ADR-0021, #37)

First run after retrieval moved from Azure AI Search to pgvector in homelab's Postgres.
Same 43 questions, same generator, same offline judge (`DeepSeek-V4-Pro`), ragas 0.4.3,
Python 3.11. The "before" numbers are the Azure-era files from f87ec34 (2026-09-13).

Only the six modes that can run without a reranker. `contextual`, `baseline` and
`raptor_sac` need the semantic ranker, which left with Azure. Their Azure-era files were
dropped in 72869fa, and they get re-evaluated once the Spark serves a reranker (it ended
up being `nvidia/llama-nemotron-rerank-1b-v2`, see the entry above).

Azure → pgvector:

| mode | hit_rate@fused | mrr@fused | faithfulness | answer_relevancy | context_precision | context_recall | abstained |
|---|---|---|---|---|---|---|---|
| graphrag | 0.951 → 0.956 | 0.576 → 0.580 | 0.806 → 0.818 | 0.781 → 0.743 | 0.450 → 0.463 | 0.393 → 0.321 | 0.349 → 0.279 |
| combined | 0.960 → 0.984 | 0.754 → 0.769 | 0.876 → 0.892 | 0.866 → 0.895 | 0.584 → 0.603 | 0.729 → 0.672 | 0.140 → 0.140 |
| baseline_agentic | 0.977 → 0.972 | 0.825 → 0.826 | 0.875 → 0.850 | 0.854 → 0.864 | 0.568 → 0.538 | 0.770 → 0.717 | 0.116 → 0.093 |
| raptor_sac_agentic | 0.974 → 0.967 | 0.857 → 0.860 | 0.896 → 0.878 | 0.862 → 0.865 | 0.571 → 0.552 | 0.742 → 0.759 | 0.047 → 0.047 |
| graphrag_agentic | 0.980 → 0.974 | 0.612 → 0.610 | 0.821 → 0.804 | 0.796 → 0.821 | 0.498 → 0.453 | 0.414 → 0.381 | 0.349 → 0.302 |
| combined_agentic | 0.985 → 0.988 | 0.769 → 0.791 | 0.859 → 0.888 | 0.830 → 0.877 | 0.578 → 0.572 | 0.591 → 0.654 | 0.163 → 0.047 |

95% bootstrap intervals on the new run:

| mode | faithfulness | answer_relevancy | context_precision | context_recall |
|---|---|---|---|---|
| graphrag | 0.74–0.89 | 0.62–0.85 | 0.35–0.57 | 0.19–0.45 |
| combined | 0.84–0.94 | 0.85–0.93 | 0.49–0.71 | 0.55–0.79 |
| baseline_agentic | 0.78–0.91 | 0.82–0.91 | 0.42–0.65 | 0.59–0.83 |
| raptor_sac_agentic | 0.84–0.92 | 0.80–0.92 | 0.43–0.67 | 0.64–0.87 |
| graphrag_agentic | 0.72–0.88 | 0.72–0.90 | 0.34–0.56 | 0.26–0.52 |
| combined_agentic | 0.86–0.92 | 0.82–0.92 | 0.46–0.68 | 0.52–0.78 |

What it says, as far as 43 questions can say anything:

- Retrieval held up. hit_rate and mrr moved by about 0.02 at most, in both directions.
- Every Azure-era judge-metric mean sits inside the new run's interval. No measurable
  regression. The only means outside it are combined_agentic's abstain rate (below) and
  combined's raptor_sac sub-leg hit rate, which went up (0.91 → 0.97).
- context_recall dipped on graphrag, combined and baseline_agentic. Best guess is the
  re-embed at import shifting which chunks come back. Not checked.
- combined_agentic abstains a lot less (16% → 5%, outside the 0.00–0.12 interval) and its
  judge scores went up anyway. The one change here that looks real.

Coverage: faithfulness and answer_relevancy are only scored on answered questions, so their
`valid` counts follow the abstain rate. Six judge jobs died across the run (three 600s
timeouts, three 429s from the DeepSeek deployment), costing at most two context_precision
scores per mode.

Run notes: on Python 3.14 the RAGAS step deadlocked at 0 jobs done (twice, hours each), so
this ran on 3.11 to match the image. Hence `.python-version`.
