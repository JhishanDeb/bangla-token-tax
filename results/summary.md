# How many tokens does Bangla cost across LLMs? -- summary

Data: FLORES-200 original NLLB release (flores200_dataset.tar.gz); N = 1012 sentence pairs
Run: 2026-10-10T06:21:02+00:00; transformers 5.16.1

## Pre-registered hypotheses

- **H1: PASS** -- Qwen2.5=5.026 [4.965, 5.086], Llama-3.1=5.841 [5.775, 5.907], Gemma-2=2.670 [2.640, 2.699]; threshold 1.5; all CI lower bounds above threshold: True
- **H2: PASS** -- lowest = BanglaBERT; Bangla fertility: BanglaBERT=1.382, mT5=2.715, Gemma-2=3.700, Qwen2.5=7.116, Llama-3.1=8.137
- **H3: PASS** -- best text ruler characters=0.988; best multilingual token ruler tokens: mT5=1.567
- **Decision rule: NOT TRIGGERED** -- at least one modern LLM has a ratio >= 1.3 (Qwen2.5=5.026 [4.965, 5.086], Llama-3.1=5.841 [5.775, 5.907], Gemma-2=2.670 [2.640, 2.699])

## Main table (corpus-level unless noted)

| tokenizer | fertility_corpus_eng | fertility_corpus_ben | fertility_mean_eng | fertility_mean_ben | tokens_per_char_eng | tokens_per_char_ben | ratio | ratio_ci_lo | ratio_ci_hi | ratio_median | ratio_q1 | ratio_q3 | ratio_note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5 | 1.261 | 7.116 | 1.271 | 7.170 | 0.209 | 1.065 | 5.026 | 4.965 | 5.086 | 5.040 | 4.439 | 5.709 |  |
| Llama-3.1 | 1.241 | 8.137 | 1.250 | 8.211 | 0.206 | 1.217 | 5.841 | 5.775 | 5.907 | 5.830 | 5.159 | 6.563 |  |
| Gemma-2 | 1.234 | 3.700 | 1.243 | 3.745 | 0.205 | 0.554 | 2.670 | 2.640 | 2.699 | 2.667 | 2.400 | 3.000 |  |
| mT5 | 1.543 | 2.715 | 1.553 | 2.740 | 0.256 | 0.406 | 1.567 | 1.553 | 1.581 | 1.567 | 1.431 | 1.714 |  |
| BanglaBERT | 2.076 | 1.382 | 2.100 | 1.395 | 0.345 | 0.207 | 0.593 | 0.587 | 0.599 | 0.583 | 0.528 | 0.657 | not meaningful (Bangla-only tokenizer) |
| BanglaBERT-raw | 2.073 | 1.381 | 2.096 | 1.394 | 0.344 | 0.207 | 0.593 | 0.587 | 0.600 | 0.583 | 0.529 | 0.657 | not meaningful (Bangla-only tokenizer) |

## Ruler comparison (Bangla/English, sorted by |ratio - 1|)

| ruler | ratio | ci_lo | ci_hi | cv | median | abs_dev_from_1 | note |
|---|---|---|---|---|---|---|---|
| characters | 0.988 | 0.980 | 0.996 | 0.140 | 0.981 | 0.012 |  |
| words | 0.891 | 0.882 | 0.900 | 0.173 | 0.889 | 0.109 |  |
| graphemes | 0.619 | 0.614 | 0.624 | 0.149 | 0.617 | 0.381 |  |
| tokens: BanglaBERT-raw | 0.593 | 0.587 | 0.600 | 0.177 | 0.583 | 0.407 | not meaningful (Bangla-only tokenizer) |
| tokens: BanglaBERT | 0.593 | 0.587 | 0.599 | 0.177 | 0.583 | 0.407 | not meaningful (Bangla-only tokenizer) |
| tokens: mT5 | 1.567 | 1.553 | 1.581 | 0.152 | 1.567 | 0.567 |  |
| bytes | 2.640 | 2.618 | 2.662 | 0.141 | 2.626 | 1.640 |  |
| tokens: Gemma-2 | 2.670 | 2.640 | 2.699 | 0.184 | 2.667 | 1.670 |  |
| tokens: Qwen2.5 | 5.026 | 4.965 | 5.086 | 0.198 | 5.040 | 4.026 |  |
| tokens: Llama-3.1 | 5.841 | 5.775 | 5.907 | 0.191 | 5.830 | 4.841 |  |

## Effective context: English-equivalent window (W / ratio)

| tokenizer | W=512 | W=2048 | W=8192 | W=32768 | W=131072 | native W | native eng-equiv | % lost | note |
|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5 | 102 | 408 | 1630 | 6520 | 26081 | 131072 | 26081 | 80 |  |
| Llama-3.1 | 88 | 351 | 1402 | 5610 | 22439 | 131072 | 22439 | 83 |  |
| Gemma-2 | 192 | 767 | 3069 | 12275 | 49099 | 8192 | 3069 | 62 |  |
| mT5 | 327 | 1307 | 5228 | 20910 | 83641 | n/a | n/a | 36 |  |
| BanglaBERT | 863 | 3454 | 13815 | 55259 | 221038 | 512 | 863 | -69 | not meaningful |
| BanglaBERT-raw | 863 | 3452 | 13806 | 55226 | 220902 | 512 | 863 | -68 | not meaningful |

## Sentences that fit (English/Bangla; "+" = whole corpus fits)

| tokenizer | W=512 | W=2048 | W=8192 | W=32768 | W=131072 | native |
|---|---|---|---|---|---|---|
| Qwen2.5 | 16/2 | 68/12 | 296/53 | 1012+/232 | 1012+/952 | 1012+/952 |
| Llama-3.1 | 16/2 | 70/11 | 304/45 | 1012+/200 | 1012+/833 | 1012+/833 |
| Gemma-2 | 16/4 | 69/22 | 303/104 | 1012+/454 | 1012+/1012+ | 303/104 |
| mT5 | 13/7 | 56/31 | 244/152 | 979/618 | 1012+/1012+ | n/a |
| BanglaBERT | 10/15 | 38/66 | 174/299 | 722/1012+ | 1012+/1012+ | 10/15 |
| BanglaBERT-raw | 10/15 | 38/66 | 174/299 | 722/1012+ | 1012+/1012+ | 10/15 |

## Diagnostics

| tokenizer | lang | n_tokens | unk_rate | partial_char_rate | roundtrip_exact_rate | bn_vocab |
|---|---|---|---|---|---|---|
| Qwen2.5 | eng | 27621 | 0.0000 | 0.0000 | 1.0000 | 47 |
| Qwen2.5 | ben | 138814 | 0.0000 | 0.6761 | 1.0000 | 47 |
| Llama-3.1 | eng | 27171 | 0.0000 | 0.0000 | 1.0000 | 9 |
| Llama-3.1 | ben | 158715 | 0.0000 | 0.8459 | 1.0000 | 9 |
| Gemma-2 | eng | 27036 | 0.0000 | 0.0000 | 1.0000 | 289 |
| Gemma-2 | ben | 72174 | 0.0000 | 0.0000 | 1.0000 | 289 |
| mT5 | eng | 33791 | 0.0000 | 0.0000 | 0.9960 | 1885 |
| mT5 | ben | 52953 | 0.0000 | 0.0000 | 0.9200 | 1885 |
| BanglaBERT | eng | 45470 | 0.0001 | 0.0000 | 0.6018 | 29127 |
| BanglaBERT | ben | 26963 | 0.0001 | 0.0000 | 0.0119 | 29127 |
| BanglaBERT-raw | eng | 45405 | 0.0017 | 0.0000 | 0.5850 | 29127 |
| BanglaBERT-raw | ben | 26941 | 0.0008 | 0.0000 | 0.0079 | 29127 |

## Skipped tokenizers

- none

## Fallbacks

- none

## Warnings

- none

Notes: BanglaBERT is a Bangla-only tokenizer; its English numbers are reported but its ratio is not meaningful. Token counts exclude special tokens (add_special_tokens=False). "Sentences that fit" sums per-sentence counts. Example tokenizations: example_tokenizations.csv.
