# Private contract assistant: evaluation report

*Generated 2026-11-04 · CUAD v1, 30 contracts, 240 questions (106 answerable, 74 unanswerable, 60 personal-data probes) · 270 planted values · retriever hybrid BM25 + dense (RRF), top-4 · seed 42*

## 1. Results

Intervals: 95% bootstrap over contracts.

| Generator | Index | Correct | Faithful citation | Says NOT FOUND when absent | False confidence | Probe leak | Ordinary-question leak | Sent off-machine |
|---|---|---|---|---|---|---|---|---|
| Extractive (no LLM) | raw | 36% [27%, 45%] | 99% [97%, 100%] | 1% [0%, 4%] | 78% [73%, 84%] | 42% [32%, 52%] | 0% [0%, 0%] | 0% [0%, 0%] |
| Extractive (no LLM) | redacted | 36% [27%, 45%] | 96% [93%, 99%] | 1% [0%, 4%] | 77% [71%, 83%] | 13% [5%, 23%] | 0% [0%, 0%] | 0% [0%, 0%] |
| Local LLM (Qwen2.5-1.5B-Instruct) | raw | 11% [6%, 18%] | 16% [9%, 24%] | 91% [83%, 97%] | 29% [22%, 36%] | 2% [0%, 5%] | 0% [0%, 0%] | 0% [0%, 0%] |
| Local LLM (Qwen2.5-1.5B-Instruct) | redacted | 9% [4%, 16%] | 15% [7%, 24%] | 91% [83%, 97%] | 29% [21%, 38%] | 0% [0%, 0%] | 1% [0%, 2%] | 0% [0%, 0%] |

## 2. Findings

* **Local LLM (Qwen2.5-1.5B-Instruct).** Told not to reveal personal data, it still leaked planted data on 2% of probe questions without redaction, and 0% with redaction (no clear difference: +2% [+0%, +5%]).
* **Redaction** removed 87% of planted values; weakest type: NAME (60%). It also redacted 0.50% of other text and altered 2% of labelled clauses.
* **Local LLM (Qwen2.5-1.5B-Instruct) − extractive (redacted), correct answer:** -26% [-37%, -16%] (a clear difference).
* **Local LLM (Qwen2.5-1.5B-Instruct) − extractive (redacted), false confidence:** -47% [-58%, -36%] (a clear difference).

## 3. Latency and cost

| Generator | Single question p50 / p95 (s) | Batch s per question | Tokens in / out | Cost per 1,000 questions |
|---|---|---|---|---|
| Extractive (no LLM) | – / – | 0.000 | 0 / 0 | $0.00 |
| Local LLM (Qwen2.5-1.5B-Instruct) | 0.56 / 3.57 | 0.608 | 1159 / 19 | not priced |

Prices: API None / None USD per million tokens in/out; GPU None USD per hour (configured, not measured).

## 4. Retrieval

Recall@4 on answerable questions: raw 87%, redacted 89%. Contracts whose index still holds planted data: raw 30/30, redacted 21/30.

## 5. Limitations

* **Planted, not natural, personal data.** The records follow three sentence templates; real personal data is messier, so redaction recall here is an upper bound for these formats and says nothing about formats not planted.
* **Names are the hard case.** Names are found by a small general-purpose NER model; a stronger model, a domain-tuned one, or a list of known people would change the name numbers.
* **Correctness is lexical.** A quote counts as correct if at least half its words lie in the labelled clause; paraphrased answers without a quote are not credited.
* **One local model, one API model, default prompts, greedy decoding.** No prompt tuning for either.
* **Probe questions are direct.** Indirect extraction attacks (role-play, multi-turn) are not tested.
* **30 contracts.** Intervals are over contracts; small differences will not be resolved.
* **Costs depend on configured prices**; latency depends on this GPU and on the API's load at run time.

## 6. Reproducibility

Python 3.13.16, spaCy 3.8.16; local model `Qwen/Qwen2.5-1.5B-Instruct`; API model `claude-haiku-5-5` (not run); seed 42; configuration in `run_config.json`.
