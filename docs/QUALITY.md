# Quality: how much of the original model the MixedK pack keeps

Measured 2026-09-03 to 2026-09-07 by [GaelicThunder](https://github.com/GaelicThunder) on one DGX Spark
(GB10, 128 GB), with this pack's tensors served through the sparkinfer route described under
[Serving path](#serving-path). Every number below has a receipt in [`receipts/`](../receipts/); the scripts are
in the sibling recipe,
[DeepSeek-V4-Flash-Vision-One-DGX-Spark](https://github.com/GaelicThunder/DeepSeek-V4-Flash-Vision-One-DGX-Spark).

## The number

The question a quantized pack has to answer is not only "how fast" but "how much of the original model is
left". The measurement here is a paired per-token log-likelihood test against the original-precision
release of the same model: 64,859 frozen tokens (wikitext prose 38,397 · gsm8k math 9,031 · Python code
17,431, in 23 chunks), scored token by token by the pack and by the original, then compared token by token.

| MixedK against the original | prose (wikitext) | math (gsm8k) | code | all tokens |
|---|---|---|---|---|
| mean NLL above the original, nats | +0.271 | +0.064 | +0.162 | +0.213 |
| paired standard error over tokens | 0.006 | 0.009 | 0.007 | 0.004 |
| **% of the original's token probability kept** | **76 %** | **94 %** | **85 %** | **81 %** |

"% kept" is exp(−Δ), the geometric mean over tokens of p_pack / p_original; 100 % would be the original
itself. With the 23 chunks as the unit (conservative, robust to correlation inside a text) the standard error
of the overall delta is 0.028 nats, z = 7.5. In perplexity terms the pack sits at 3.62 against the original's
2.93 on this corpus (wikitext 5.08 against 3.88).

Reading: 2-bit trellis on 37 layers with **all 256 experts kept** costs about a quarter of the original's
per-token probability on prose, a sixth on code and 6 % on grade-school math. The other single-Spark route
for this model, pruning 40 experts and spending 3 bits on the 216 that remain, measures 57 / 98 / 94 / 70 %
on the same tokens: better on code, far worse on prose, because the pruned experts are the ones rare words
and names need. That decomposition, done with byte-identical expert tensors on both sides, is in the sibling
recipe's
[`docs/KALIBRATED.md`](https://github.com/GaelicThunder/DeepSeek-V4-Flash-Vision-One-DGX-Spark/blob/main/docs/KALIBRATED.md#3-experts-versus-bits-with-the-same-weights).

## Method

- **Corpus.** `scripts/nll_corpus.py` (sibling recipe) takes fixed, public, ungated Hub datasets in their
  published order, no shuffling: wikitext-103 test, gsm8k test, CodeXGLUE code-to-text Python test. Chunks
  are cut on paragraph or item boundaries with the model's tokenizer so each one prefills in one pass
  (≤ 3,072 tokens). Every engine tokenizes the same text with the same tokenizer, so the token sequence is
  identical across packs.
- **Scoring.** Each chunk goes to `/v1/completions` with `max_tokens: 1` and `prompt_logprobs: 0`; the
  engine returns the log-probability of every prompt token under the model. This is the prefill path: no
  sampling, no temperature, reproducible to the third decimal across boots.
- **Pairing.** The per-token values are stored (that is what the 550 KB receipts are), so two engines are
  compared on exactly the same tokens and the text's own difficulty cancels; what remains is the
  difference between the models. `scripts/nll_probe.py --compare A.json B.json` prints the delta with its
  token-level and chunk-level standard errors.
- **Reference.** The abliterated Vision-Exp source this pack was converted from
  ([drowzeys/keys-DeepSeekV4Flash-Vision-EXP-ablit](https://huggingface.co/drowzeys/keys-DeepSeekV4Flash-Vision-EXP-ablit),
  157 GB, the release precision: FP8 attention, MXFP4 experts), served with vLLM 0.28.1 nightly at
  tensor-parallel 2 on a 2×H200 pod and scored with the same script on the same tokens on 2026-09-06
  (`receipts/nll-reference-fp8-20260906.json`). "The original" is therefore the abliterated model at full
  precision, and the deltas isolate the quantization.

## Accuracy gates

Run on the pack twice, on two separate boots (2026-09-03), which gives the harness's own run-to-run noise.

| | first boot | second boot |
|---|---|---|
| MMLU-Pro, 251 items, options in original order | 161/251 · 64.1 % | 161/251 |
| MMLU-Pro, same items, options rotated by 3 | 162/251 · 64.5 % | 162/251 |
| **mean** | **64.3 %** | **64.3 %** |
| correct under both orders · agreement between orders | 134 · 78.1 % | 133 · 77.3 % |
| MATH-500 level 5, 60 items, `\boxed{}` exact match, greedy | 50/60 (3 truncated at 2,400 tokens) | 49/60 (5 truncated) |
| needle, 3 depths × 4k / 32k / 131k tokens | 9/9 | 9/9 |
| perplexity, 8 fixed passages (Italian and English prose, technical prose, Python, bash) | 4.545 | 4.543 |

MMLU-Pro (TIGER-Lab, 18 items per category, seed fixed) is scored on the log-probability of the answer letter
after `…\nAnswer:`, one forward pass per item, so no item is lost to a token budget; asking each item in two
option orders controls position bias. MATH is free generation with thinking off, `max_tokens 2400`,
temperature 0. The needle filler is repetitive and not salted, so the prefix cache makes its timings
meaningless; the pass/fail stands. Receipts: `receipts/bench-mixedk-20260903.json`,
`receipts/bench-mixedk-2nd-boot-20260903.json`, `receipts/ppl-mixedk-20260903.json`.

For orientation only: the other single-Spark pack people run beside this one, the 216-expert 3-bit REAP pack
of DeepSeek-V4-Flash-**0731** (the text release, so a *different base model*), scores 60.8 % on the same 251
items (exact McNemar on the 94 discordant decisions: p = 0.079), 50/60 on MATH, 9/9 on the needle and 5.373
on the perplexity passages
([receipts and per-category table](https://github.com/GaelicThunder/DeepSeek-V4-Flash-Vision-One-DGX-Spark/blob/main/docs/BENCHMARK.md)).
The 2-bit pack with all experts kept is not less capable than the 3-bit pack with 40 experts pruned; "not
worse" is the safe statement.

## Serving path

These measurements did not run on the vllm-exl3 route this repository documents. They ran on the sibling
recipe's route: the sparkinfer image (0xSero's vLLM fork build) with the pack converted to its rank-sliced
tp1 layout. What that means for reading the numbers:

- the expert trellis tensors are this pack's, copied unchanged (the conversion verifies a sample against the
  source shards), with the same six layers at 3-bit;
- the non-routed tensors (attention, indexer, shared experts, dense layers, head) are quantized to 128×128
  block-FP8 at load on that route, where this repository's recommended route keeps them BF16 as stored, so
  if anything the vllm-exl3 route should score equal or marginally better;
- the KV cache is the image's packed NVFP4 MLA format;
- the DSpark draft was built from the pack's own `mtp.*` tensors with 64 of the 256 draft experts;
  speculative decoding does not change prefill likelihoods, so it plays no part in the numbers above;
- `max_model_len 245760`, one sequence at a time.

Likelihood is a property of the weights, not of the scheduler, so the retention numbers should reproduce
closely on this repository's route; they have not been run on it. Speed was left out of this page on
purpose: it belongs to the stack, not to the pack.

## What these numbers do not tell you

- The likelihood test is teacher-forced (prefill). It measures how closely the pack tracks the original's
  next-token distribution on fixed text, which is the right instrument for ranking quantizations of one
  model; it is not an end-to-end generation score and not comparable with any leaderboard.
- MMLU-Pro on 251 items has a sampling error of about ±3 points, and letter-logprob scoring is not the
  chain-of-thought protocol published numbers use. Differences of a few points between two packs of the same
  model are ties; the paired likelihood is what resolves them.
- MATH-500 on 60 items is a gate, not a ranking. The needle is a pass/fail.
- The corpus is English prose, grade-school math and Python; other languages and long-context behaviour
  are not covered.
- One serving stack, one machine, one operator.

## Reproducing

From the sibling recipe, against any OpenAI-compatible endpoint serving this pack (here the port and model
name of `scripts/serve_one_spark_dsv4.sh`):

```bash
git clone https://github.com/GaelicThunder/DeepSeek-V4-Flash-Vision-One-DGX-Spark
cd DeepSeek-V4-Flash-Vision-One-DGX-Spark
pip install datasets

python3 scripts/nll_corpus.py nll_corpus_60k.jsonl --tokenizer ~/models/DSV4-Flash-Vision-ablit-EXL3-MixedK
python3 scripts/nll_probe.py nll_corpus_60k.jsonl out.json --base http://127.0.0.1:8899 --model DSV4-Flash
python3 scripts/nll_probe.py --compare receipts/nll/nll-ref-source-fp8-20260906.json out.json

python3 scripts/bench/prep_data.py
python3 scripts/bench/bench_all.py bench.json --base http://127.0.0.1:8899 --only mc,math,needle
python3 scripts/ppl_probe.py http://127.0.0.1:8899 DSV4-Flash ppl.json
```

The corpus is rebuilt deterministically from the public datasets (fixed splits, published order, boundaries
cut with the pack's tokenizer), so a fresh run is paired with the receipts here token for token.
