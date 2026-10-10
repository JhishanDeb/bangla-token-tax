# How many tokens does Bangla cost across LLMs?

`bangla_token_tax.ipynb` is a self-contained Kaggle notebook. It measures how many more tokens Bangla needs than English for the same meaning, using FLORES-200 devtest (1,012 aligned sentences), and compares five tokenizers with four tokenizer-free "rulers". It downloads only tokenizers and configs, never model weights. It does no training and needs no GPU.

## Kaggle settings

| Setting | Value |
|---|---|
| Accelerator | **None** (CPU) |
| Internet | **On** (Settings → Internet) |
| Environment | Default image. Tested with transformers 4.46.3, which is the version installed if transformers is missing. |
| Run | **Save Version → Save & Run All** |

## Secrets

Add-ons → Secrets → add `HF_TOKEN` (a Hugging Face read token) and attach it to the notebook.

The token is used for these gated repos:
- `meta-llama/Llama-3.1-8B` and `google/gemma-2-9b`. Accept their licences on Hugging Face first. Without access, the notebook falls back to `unsloth/Meta-Llama-3.1-8B` and `unsloth/gemma-2-9b`, and it logs that the mirror's vocab could not be checked against the official repo.
- `openlanguagedata/flores_plus`. This is only the second data source, used if the NLLB tarball fails.

If the secret is missing, the notebook logs a warning and keeps going with the fallbacks.

## Packages

These are pip-installed only if they are missing: `sentencepiece`, `protobuf`, `regex`, and the BanglaBERT normalizer (`git+https://github.com/csebuetnlp/normalizer`). If transformers cannot be imported, `transformers==4.46.3` is installed.

## Expected runtime

About 2–5 minutes on Kaggle CPU. Most of that is downloads: the 25 MB FLORES tarball, about 60 MB of tokenizer files, and the normalizer install. The compute part (tokenization, a 10,000-resample paired bootstrap, and decoding every vocab id) took about 25 s on a laptop.

## Data source order

1. `https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz`, original FLORES-200. The SHA-256 is recorded.
2. `openlanguagedata/flores_plus` (the later corrected FLORES+), pinned to the dataset commit. The `.parquet` or `.jsonl` files are inspected and their columns logged.
3. Otherwise the notebook stops and asks you to attach `eng_Latn.devtest` and `ben_Beng.devtest` as a Kaggle dataset.

`run_info.json → data.version` records which version was used.

## Output files (`/kaggle/working/`)

| File | Contents |
|---|---|
| `results_main.csv` | One row per tokenizer: repo, revision, class, vocab size, native window, fertility (corpus and mean), tokens/char, Bangla/English ratio with 95% CI, median/IQR, CV |
| `ruler_comparison.csv` | Words, characters, graphemes, bytes and every tokenizer: ratio, CI, CV, distance from 1 |
| `effective_context.csv` | W ∈ {512, 2k, 8k, 32k, 128k} plus native window: English-equivalent window, % lost, sentences that fit (en/bn), corpus-exhausted flags |
| `per_sentence.csv` | Texts and every ruler count per sentence, plus per-sentence ratios |
| `example_tokenizations.csv` | Token pieces (raw and decoded) for 3 fixed sentences × tokenizer × language |
| `diagnostics.json` | UNK rate, partial-character (U+FFFD) rate, Bengali-block vocab count, round-trip rate, ZWJ/ZWNJ and NFC counts |
| `run_info.json` | Date, Python and library versions, data source/version, tokenizer repo/revision/vocab, fallbacks, warnings, hypothesis verdicts, runtime |
| `summary.md` | All tables, hypothesis verdicts, skipped tokenizers, fallbacks and warnings |
| `figures/fertility.png` | Tokens per word, English and Bangla side by side |
| `figures/token_ratio.png` | Bangla/English token ratio with 95% CI, parity line at 1 |
| `figures/ruler_comparison.png` | Ratio per ruler with CI and CV |
| `figures/ratio_distribution.png` | Violin plot of per-sentence ratios per tokenizer |
| `figures/effective_context.png` | English-equivalent window vs W (star = native window) |

The last cell prints a block between `===== SHARE WITH CLAUDE: START =====` and `===== SHARE WITH CLAUDE: END =====` (about 120 lines). Copy it back into the chat.

## Methodological notes

- Text is Unicode NFC-normalized. In FLORES-200 Bangla, NFC decomposes the precomposed letters য় U+09DF, ড় U+09DC and ঢ় U+09DD into a base letter plus nukta, because they are Unicode composition exclusions. This changes 609 of 1,012 Bangla sentences and adds about 1% more code points. `diagnostics.json` records the counts.
- Token counts use `add_special_tokens=False`.
- BanglaBERT appears twice: with the csebuetnlp normalizer (main table) and without it (`BanglaBERT-raw`). It is Bangla-only, so its Bangla/English ratio is labeled "not meaningful".
- The bootstrap is paired. One 10,000 × N index matrix (seed 0) is shared by every ruler, so their CIs can be compared directly.
- "Sentences that fit" adds up per-sentence token counts in FLORES order. A `+` means the whole corpus fits, so the count is a lower bound.
- Local smoke test only: setting the environment variable `BTT_SMOKE_N=20` runs the notebook on the first 20 sentences (after the full-corpus checks). Leave it unset on Kaggle.

`src/` contains the percent-format source (`bangla_token_tax.py`) and `build_notebook.py`, which regenerates the `.ipynb` from it.
