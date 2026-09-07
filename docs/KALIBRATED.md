# Kalibrated Vision Exp: this pack with 22 more expert layers at a calibrated 3-bit

**Kalibrated Vision Exp** (short: Kalibrated) is the MixedK pack with 22 more expert layers re-quantized at a
*calibrated* 3-bit, so that 28 of the 43 expert layers are 3-bit instead of 6, with all 256 routed experts
kept in every layer. Everything else, the other 15 expert layers, attention, indexer, shared experts, router
rows and hash tables, head, vision tower, aligner and the three DSpark MTP layers, is this pack's files,
unchanged. The K in the name is MixedK's; the "calibrated" is the part added. Built and measured 2026-09-06/07
by [GaelicThunder](https://github.com/GaelicThunder); published at
[GaelicThunder/DeepSeek-V4-Flash-Vision-Exp-ablit-EXL3-Kalibrated](https://huggingface.co/GaelicThunder/DeepSeek-V4-Flash-Vision-Exp-ablit-EXL3-Kalibrated)
(public, not gated, 106 GB) and served by the sibling recipe
[DeepSeek-V4-Flash-Vision-One-DGX-Spark](https://github.com/GaelicThunder/DeepSeek-V4-Flash-Vision-One-DGX-Spark).

The added tensors are a re-conversion, from the same abliterated source, of layers this pack already ships,
in the same trellis format and codebook, made so they drop into MixedK's layout. If the layers are wanted in
MixedK itself they are there to be taken, same format, same layout. Following this repository's rule, nothing
here describes how they were produced; that is documented in the sibling recipe's
[`docs/KALIBRATED.md`](https://github.com/GaelicThunder/DeepSeek-V4-Flash-Vision-One-DGX-Spark/blob/main/docs/KALIBRATED.md)
and reproducible from its
[`scripts/kalibrated/`](https://github.com/GaelicThunder/DeepSeek-V4-Flash-Vision-One-DGX-Spark/tree/main/scripts/kalibrated).

## What changes, tensor by tensor

| | MixedK | Kalibrated Vision Exp |
|---|---|---|
| routed experts per layer | 256 | 256 |
| expert layers at 3-bit | 6: `3 13 21 22 28 41` | 28: the same six plus `27 23 31 35 32 34 37 40 1 39 25 29 8 30 19 26 24 11 12 9 0 42` |
| expert layers at 2-bit | 37 | 15 |
| codebook, trellis format | `mcg`, EXL3 | same |
| non-routed tensors, vision tower, aligner, `mtp.*`, tokenizer | | identical files |
| weights resident on the sparkinfer route ([below](#where-the-memory-comes-from)) | 83.6 GiB | 100.1 GiB (+22 × 0.75 GiB) |

The 22 layers are listed in ranking order (see [Which layers, and why](#which-layers-and-why)). Promoting
one layer costs 0.75 GiB (256 experts × 3 MiB). 26 promoted layers still load on a 128 GB GB10 but leave
0.63 GiB of KV pool against the 3.91 GiB one 245,760-token request needs on that route, so the published pack
stops at 22.

## Results, paired on the same tokens

Same method as [`QUALITY.md`](QUALITY.md): 64,859 frozen tokens, prefill log-likelihood, paired token by
token, the original being the abliterated Vision-Exp source at release precision.

| % of the original's token probability kept | prose (wikitext) | math (gsm8k) | code | all |
|---|---|---|---|---|
| **Kalibrated Vision Exp** | **87 %** | **99.8 %** | **92 %** | **90 %** |
| MixedK | 76 % | 94 % | 85 % | 81 % |

| Kalibrated − MixedK | prose | math | code | all |
|---|---|---|---|---|
| Δ nats (negative = Kalibrated closer to the original) | **−0.133** | **−0.062** | **−0.081** | **−0.109** |
| paired standard error over tokens | 0.005 | 0.008 | 0.005 | 0.0035 |
| standard error with the 23 chunks as units | | | | 0.014 |
| as perplexity | −12.5 % | −6.0 % | −7.8 % | −10.4 % |

Receipts: `receipts/nll-mixedk-20260905.json`, `receipts/nll-kalibrated-20260907.json`,
`receipts/nll-reference-fp8-20260906.json`.

The other gates move within their noise. MMLU-Pro 63.2 % against 64.3 % (159 + 158 against 161 + 162 of 502
decisions: three items per order, inside the ±3-point sampling error of a 251-item set;
`receipts/bench-kalibrated-20260907.json`). Perplexity on the 8 fixed passages 4.487 against 4.545
(`receipts/ppl-kalibrated-20260907.json`). Two- and three-image prompts answer correctly per image
([log](https://github.com/GaelicThunder/DeepSeek-V4-Flash-Vision-One-DGX-Spark/blob/main/receipts/kalibrated/multi_image.log)).
The paired likelihood is the measurement that separates the two packs; the gates say only that nothing broke.

## Which layers, and why

The conversion log of a 3-bit build of this same model reports, for every expert tensor, the error of the
quantized weight on the calibration activations (`proxy_err`). Averaged over the 768 expert tensors of a
layer it says how much that layer amplifies quantization error. Kalibrated promotes the top of that ranking,
skipping the six layers MixedK already holds at 3-bit (they sit in the middle of the same ranking).

The ranking was then checked by measurement rather than assumed: three one-boot trials, each the served
22-layer set with four layers swapped for four spare 3-bit conversions (`33 36 16 6`), scored on the same
64,859 tokens against the served set:

| swapped out for the spares | Δ nats, all tokens | prose | math | code | z (token SE) |
|---|---|---|---|---|---|
| the four lowest-ranked promoted layers, `12 9 0 42` | −0.001 | −0.001 | −0.003 | −0.001 | −0.7 (noise) |
| four mid-ranked layers, `25 29 8 30` | +0.011 | +0.014 | +0.004 | +0.008 | +4.5 |
| the top four, `27 23 31 35` | +0.017 | +0.025 | +0.011 | +0.002 | +6.9 |

The head of the ranking carries the gain, the tail is interchangeable with the spares, and the served set is
the best of the four measured. Under a linear share of the measured bits gain the 22 layers were expected to
give about −0.05 nats on prose; they gave −0.133. Receipts for the three trials:
[`receipts/nll/nll-swap-{tail,mid,head}-20260907.json`](https://github.com/GaelicThunder/DeepSeek-V4-Flash-Vision-One-DGX-Spark/tree/main/receipts/nll)
in the sibling recipe.

## Where the memory comes from

On the sibling recipe's route (the sparkinfer image: non-routed tensors quantized to block-FP8 at load, CUDA
graphs on) MixedK is resident at 83.6 GiB and leaves a 986k-token KV pool at utilization 0.88. Kalibrated
spends 16.5 GiB of that on expert bits instead: 100.1 GiB resident, a 269k-token KV pool at 0.925, one
request of 245,760 tokens still fits. The cost per decode step is the extra bytes read: on that route the
verify step runs at 10.4–10.5 per second against 10.9–11.1 for MixedK (about 5 % slower), and tokens per
second land within noise of MixedK's because the closer model accepts slightly more draft tokens
([dsbench receipts](https://github.com/GaelicThunder/DeepSeek-V4-Flash-Vision-One-DGX-Spark/tree/main/receipts/kalibrated)).

On this repository's recommended route the non-routed tensors stay BF16 as stored, so the same 16.5 GiB
would have to come out of the KV pool; whether a useful context remains has not been tried. Nothing in this
is a shortcoming of MixedK: it is the same pack under a different memory budget.

## Running it

The Hub repository is in the rank-sliced tp1 layout the sparkinfer image reads (`tp1/` with a
`rank-sliced-tp1-manifest.json`, plus `dspark-draft-k64/`); it is what the sibling recipe's `start.sh`
downloads and serves. It has not been tried on the vllm-exl3 route. The tensors carry the same names and
formats as MixedK's, so a repack into this repository's 48-shard layout should serve here with no code
change: `scripts/fix_pack_config.py` scans the trellis widths and would write the 28 `layer_bits` entries by
itself. That repack is straightforward and, as of this page, untested.

## Receipts

Listed in [`receipts/README.md`](../receipts/README.md), with the mapping to the sibling recipe's file names.
