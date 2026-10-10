# %% [markdown]
# # How many tokens does Bangla cost across LLMs?
#
# Motivation measurement for a position-encoding paper. ALiBi and RoPE count distance in **tokens**.
# If Bangla needs many more tokens than English for the same meaning, the same real distance looks
# longer in Bangla and less text fits in the context window. This notebook measures that gap on
# FLORES-200 devtest (1,012 aligned English/Bangla sentences) for five tokenizers, and compares
# alternative "rulers" (words, characters, grapheme clusters, bytes) to see which is most equal
# across the two languages. No model weights are downloaded; only tokenizers and configs.
#
# ## Pre-registered expectations
# *Written before any results were computed. This cell is never edited afterwards.*
#
# - **H1:** Llama 3.1, Qwen 2.5 and Gemma 2 each have a Bangla/English token ratio > 1.5.
# - **H2:** BanglaBERT has the lowest Bangla fertility of the five.
# - **H3:** characters or graphemes give a Bangla/English ratio closer to 1 than any tokenizer's tokens.
# - **Decision rule:** if all three modern LLMs have ratios < 1.3, the position-encoding motivation is
#   weak for current models, and the summary must say so plainly.
#
# Operational definitions (fixed in advance): "ratio" is the corpus-level ratio
# (sum of Bangla tokens / sum of English tokens). "Fertility" in H2 is corpus-level tokens per
# whitespace word, using the normalized BanglaBERT variant. In H3, "any tokenizer" means the four
# multilingual tokenizers (BanglaBERT's ratio is not meaningful because it is Bangla-only), and
# "closer to 1" means smallest |ratio - 1|.

# %% [markdown]
# ## 1. Setup: install only what is missing, import, prepare output folders

# %%
import os, sys, time, json, math, shutil, hashlib, subprocess, importlib, platform, datetime
import tarfile, tempfile, urllib.request, unicodedata

T0 = time.time()
ON_KAGGLE = os.path.isdir('/kaggle/working')
OUT = '/kaggle/working' if ON_KAGGLE else os.path.abspath(os.environ.get('BTT_OUT', 'working'))
FIG = os.path.join(OUT, 'figures')
os.makedirs(FIG, exist_ok=True)
# BTT_SMOKE_N is only for local smoke tests. On Kaggle it is unset, so the full corpus is used.
SMOKE_N = int(os.environ.get('BTT_SMOKE_N', '0'))

WARNINGS, FALLBACKS, SKIPPED = [], [], []
def warn(msg):
    WARNINGS.append(msg); print('WARNING:', msg)
def fallback(msg):
    FALLBACKS.append(msg); print('FALLBACK:', msg)

def _missing(mod):
    try:
        importlib.import_module(mod); return False
    except Exception:
        return True

PIP_IF_MISSING = {
    'sentencepiece': 'sentencepiece',
    'google.protobuf': 'protobuf',
    'regex': 'regex',
    'normalizer': 'git+https://github.com/csebuetnlp/normalizer',
}
need = [pkg for mod, pkg in PIP_IF_MISSING.items() if _missing(mod)]
if need:
    print('pip installing:', need)
    r = subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', *need], capture_output=True, text=True)
    if r.returncode != 0:
        warn(f'pip install exited with {r.returncode}: {r.stderr.strip()[-400:]}')
    importlib.invalidate_caches()
else:
    print('All optional packages already installed.')

try:
    import transformers
except Exception as e:
    warn(f'transformers import failed ({e}); installing transformers==4.46.3')
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'transformers==4.46.3'], check=True)
    importlib.invalidate_caches()
    import transformers

import numpy as np
import pandas as pd
import regex
import matplotlib
import matplotlib.pyplot as plt
import huggingface_hub
from huggingface_hub import HfApi, hf_hub_download
from transformers import AutoTokenizer, AutoConfig
transformers.logging.set_verbosity_error()

try:
    from normalizer import normalize as bn_normalize
except Exception as e:
    bn_normalize = None
    warn(f'csebuetnlp normalizer unavailable ({type(e).__name__}: {e}); BanglaBERT normalized variant will be skipped')

pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 40)
print(f'Python {sys.version.split()[0]} | transformers {transformers.__version__} | '
      f'huggingface_hub {huggingface_hub.__version__} | numpy {np.__version__} | pandas {pd.__version__}')
print('Output folder:', OUT, '| smoke test N =', SMOKE_N or 'off (full run)')

# %% [markdown]
# ## 2. Hugging Face token from Kaggle Secrets

# %%
HF_TOKEN = None
try:
    from kaggle_secrets import UserSecretsClient
    HF_TOKEN = UserSecretsClient().get_secret('HF_TOKEN')
except Exception as e:
    HF_TOKEN = os.environ.get('HF_TOKEN') or None
    if HF_TOKEN is None:
        warn(f'HF_TOKEN not available ({type(e).__name__}); gated repos will fail and fallbacks will be used')
if HF_TOKEN:
    os.environ['HF_TOKEN'] = HF_TOKEN
print('HF_TOKEN present:', bool(HF_TOKEN))
api = HfApi(token=HF_TOKEN)

# %% [markdown]
# ## 3. Data: FLORES-200 devtest, eng_Latn + ben_Beng
# Source order: (1) NLLB tarball, (2) HF `openlanguagedata/flores_plus` (gated; a later corrected
# version), (3) stop and ask for the files to be attached as a Kaggle dataset.

# %%
N_EXPECTED = 1012
LANGS = ('eng_Latn', 'ben_Beng')
DATA_INFO = {'attempts': []}

def _split_lines(raw):
    lines = raw.split('\n')
    if lines and lines[-1] == '':          # exactly one trailing newline is allowed
        lines = lines[:-1]
    return [l[:-1] if l.endswith('\r') else l for l in lines]

def load_nllb_tarball():
    url = 'https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz'
    tmp = tempfile.mkdtemp(prefix='flores_')
    path = os.path.join(tmp, 'flores200_dataset.tar.gz')
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=300) as r, open(path, 'wb') as f:
        shutil.copyfileobj(r, f)
    sha = hashlib.sha256(open(path, 'rb').read()).hexdigest()
    out = {}
    with tarfile.open(path, 'r:gz') as tf:
        names = tf.getnames()
        for lang in LANGS:
            want = f'flores200_dataset/devtest/{lang}.devtest'
            hits = [n for n in names if n.lstrip('./') == want]     # members may carry a './' prefix
            if len(hits) != 1:
                raise KeyError(f'{want} not found exactly once in tarball (found {hits})')
            out[lang] = _split_lines(tf.extractfile(hits[0]).read().decode('utf-8'))
    shutil.rmtree(tmp, ignore_errors=True)
    return out, {'source': url, 'version': 'FLORES-200 original NLLB release (flores200_dataset.tar.gz)',
                 'tarball_sha256': sha, 'order_key': 'line order in .devtest files'}

def load_flores_plus():
    repo = 'openlanguagedata/flores_plus'
    info = api.dataset_info(repo, files_metadata=False)
    files = [s.rfilename for s in info.siblings]
    out, ids, meta = {}, {}, {'source': f'hf://datasets/{repo}', 'revision': info.sha}
    for lang in LANGS:
        cands = [f for f in files if f.endswith(('.parquet', '.jsonl')) and 'devtest' in f and lang in f]
        if len(cands) != 1:
            raise RuntimeError(f'expected exactly one devtest .parquet/.jsonl for {lang}, found {cands}')
        local = hf_hub_download(repo, cands[0], repo_type='dataset', revision=info.sha, token=HF_TOKEN)
        df = pd.read_parquet(local) if local.endswith('.parquet') else pd.read_json(local, lines=True)
        print(f'  {lang}: file={cands[0]} columns={list(df.columns)}')
        meta[f'columns_{lang}'] = list(df.columns)
        text_col = next((c for c in ('text', 'sentence') if c in df.columns), None)
        if text_col is None:
            raise RuntimeError(f'no text/sentence column in {cands[0]}; columns are {list(df.columns)}')
        if 'id' in df.columns:
            df = df.sort_values('id', kind='stable')
            ids[lang] = df['id'].tolist()
        out[lang] = df[text_col].astype(str).tolist()
        if 'last_updated' in df.columns:
            meta[f'last_updated_{lang}'] = sorted(map(str, df['last_updated'].unique()))[-3:]
    if len(ids) == 2:
        if ids['eng_Latn'] != ids['ben_Beng']:
            raise RuntimeError('FLORES+ ids differ between eng_Latn and ben_Beng: sentences are not aligned')
        meta['order_key'] = 'FLORES+ id column (identical across languages)'
    else:
        meta['order_key'] = 'file row order (no id column found)'
    meta['version'] = f'FLORES+ (HF {repo} @ {info.sha})'
    return out, meta

RAW = None
for loader in (load_nllb_tarball, load_flores_plus):
    try:
        print(f'Trying {loader.__name__} ...')
        RAW, meta = loader()
        DATA_INFO.update(meta); DATA_INFO['attempts'].append(f'{loader.__name__}: OK')
        break
    except Exception as e:
        DATA_INFO['attempts'].append(f'{loader.__name__}: {type(e).__name__}: {str(e)[:300]}')
        print('  failed:', DATA_INFO['attempts'][-1])
if RAW is None:
    raise RuntimeError(
        'Could not load FLORES-200 devtest from any source.\n'
        + '\n'.join(DATA_INFO['attempts'])
        + '\nPlease attach eng_Latn.devtest and ben_Beng.devtest as a Kaggle dataset and point the loader at them.')
if not DATA_INFO['attempts'][0].endswith('OK'):
    fallback(f"data: primary source failed, used {DATA_INFO['version']}")
print('DATA SOURCE USED:', DATA_INFO['version'])

# %% [markdown]
# ## 4. Data checks, NFC, ZWJ/ZWNJ

# %%
for lang in LANGS:
    n = len(RAW[lang])
    assert n == N_EXPECTED, f'{lang}: expected {N_EXPECTED} lines, got {n}'
    empty = [i for i, s in enumerate(RAW[lang]) if not s.strip()]
    assert not empty, f'{lang}: empty lines at indices {empty[:10]}'

# Order check: numbers in a sentence should survive translation. Compare aligned pairs against the
# same pairs shifted by one; aligned agreement must be clearly higher, otherwise the files are misaligned.
_BN_DIGITS = str.maketrans('০১২৩৪৫৬৭৮৯', '0123456789')
def _nums(s):
    return sorted(regex.findall(r'\d+', s.translate(_BN_DIGITS).replace(',', '')))
def _num_agreement(en, bn):
    pairs = [(a, b) for a, b in zip(en, bn) if _nums(a)]
    return sum(_nums(a) == _nums(b) for a, b in pairs) / max(len(pairs), 1), len(pairs)
agree_aligned, n_num = _num_agreement(RAW['eng_Latn'], RAW['ben_Beng'])
agree_shifted, _ = _num_agreement(RAW['eng_Latn'], RAW['ben_Beng'][1:] + RAW['ben_Beng'][:1])
print(f'Order check on {n_num} sentences with digits: aligned agreement={agree_aligned:.3f}, shifted-by-one={agree_shifted:.3f}')
assert agree_aligned > agree_shifted + 0.2, 'English/Bangla files look misaligned (number agreement no better than shifted)'
DATA_INFO['order_check'] = {'n_with_digits': n_num, 'aligned_agreement': agree_aligned, 'shifted_agreement': agree_shifted}

ZW = {'ZWJ': '‍', 'ZWNJ': '‌'}
zw_before = {k: sum(s.count(c) for s in RAW['ben_Beng']) for k, c in ZW.items()}
EN = [unicodedata.normalize('NFC', s) for s in RAW['eng_Latn']]
BN = [unicodedata.normalize('NFC', s) for s in RAW['ben_Beng']]
zw_after = {k: sum(s.count(c) for s in BN) for k, c in ZW.items()}
nfc_changed = {'eng': sum(a != b for a, b in zip(RAW['eng_Latn'], EN)),
               'ben': sum(a != b for a, b in zip(RAW['ben_Beng'], BN))}
fffd_in_source = sum(s.count('�') for s in EN + BN)
if fffd_in_source:
    warn(f'source text already contains {fffd_in_source} U+FFFD characters; partial-character rate is inflated')
DATA_INFO.update({'zw_counts_ben_before_nfc': zw_before, 'zw_counts_ben_after_nfc': zw_after,
                  'sentences_changed_by_nfc': nfc_changed, 'n_sentences_loaded': N_EXPECTED})
print('Bangla ZWJ/ZWNJ counts before NFC:', zw_before, '| after NFC:', zw_after)
print('Sentences changed by NFC:', nfc_changed)
print('\nFirst pair (eye check):\n  EN:', EN[0], '\n  BN:', BN[0])

if SMOKE_N:
    EN, BN = EN[:SMOKE_N], BN[:SMOKE_N]
    warn(f'SMOKE TEST: using only the first {SMOKE_N} sentences. Numbers are not the real results.')
N = len(EN)
EX_IDX = sorted({0, N // 2, N - 1})       # 3 fixed example sentences: first, middle, last
print('N sentences used:', N, '| example indices:', EX_IDX)

# %% [markdown]
# ## 5. Tokenizers (tokenizer + config only, pinned to a commit hash)

# %%
TOK_SPECS = [
    ('Qwen2.5',    ['Qwen/Qwen2.5-7B'],                                    'llm'),
    ('Llama-3.1',  ['meta-llama/Llama-3.1-8B', 'unsloth/Meta-Llama-3.1-8B'], 'llm'),
    ('Gemma-2',    ['google/gemma-2-9b', 'unsloth/gemma-2-9b'],              'llm'),
    ('mT5',        ['google/mt5-base'],                                    'multilingual'),
    ('BanglaBERT', ['csebuetnlp/banglabert'],                              'bangla_only'),
]
LLM_NAMES = ['Qwen2.5', 'Llama-3.1', 'Gemma-2']
MULTI_NAMES = LLM_NAMES + ['mT5']

def _load(repo):
    sha = api.model_info(repo).sha
    tok = AutoTokenizer.from_pretrained(repo, revision=sha, token=HF_TOKEN)
    native = None
    try:
        cfg = AutoConfig.from_pretrained(repo, revision=sha, token=HF_TOKEN)
        native = getattr(cfg, 'max_position_embeddings', None)
    except Exception as e:
        warn(f'{repo}: AutoConfig failed ({type(e).__name__}); native window = n/a')
    return tok, sha, native

TOKS = {}
for name, repos, role in TOK_SPECS:
    loaded, errors = None, []
    for repo in repos:
        try:
            loaded = (repo, *_load(repo)); break
        except Exception as e:
            errors.append(f'{repo}: {type(e).__name__}: {str(e).splitlines()[0][:160]}')
    if loaded is None:
        SKIPPED.append(f'{name}: ' + ' || '.join(errors)); warn(f'{name} could not be loaded; skipped. {errors}')
        continue
    repo, tok, sha, native = loaded
    mirror_check = None
    if repo != repos[0]:
        fallback(f'{name}: official repo {repos[0]} not loadable ({errors[0]}); used mirror {repo}')
        mirror_check = 'official repo not reachable, so vocab identity with the mirror could not be verified'
        warn(f'{name}: {mirror_check}')
    elif len(repos) > 1:
        try:
            mtok = AutoTokenizer.from_pretrained(repos[1], revision=api.model_info(repos[1]).sha, token=HF_TOKEN)
            same = mtok.get_vocab() == tok.get_vocab()
            mirror_check = f'mirror {repos[1]} vocab identical to official: {same}'
            if not same:
                warn(f'{name}: {mirror_check}')
        except Exception as e:
            mirror_check = f'mirror {repos[1]} not checked ({type(e).__name__})'
    base = dict(repo=repo, revision=sha, tokenizer_class=type(tok).__name__, vocab_size=tok.vocab_size,
                vocab_len_incl_added=len(tok), native_window=native, role=role, tok=tok,
                fallback_used=repo != repos[0], mirror_check=mirror_check, pre=None)
    if name == 'BanglaBERT':
        if bn_normalize is not None:
            TOKS['BanglaBERT'] = dict(base, pre=bn_normalize, variant='csebuetnlp normalizer applied (main)')
        TOKS['BanglaBERT-raw'] = dict(base, variant='no normalizer')
    else:
        TOKS[name] = dict(base, variant='')
    print(f"{name:12s} {repo} @ {sha[:10]} {type(tok).__name__} vocab={tok.vocab_size} "
          f"native={native if native is not None else 'n/a'} {mirror_check or ''}")

if not TOKS:
    raise RuntimeError('No tokenizer could be loaded. Check internet and HF_TOKEN.')
NOT_MEANINGFUL = {'BanglaBERT', 'BanglaBERT-raw'}
MAIN_TOKS = [n for n in TOKS if not (n == 'BanglaBERT-raw' and 'BanglaBERT' in TOKS)]
if 'BanglaBERT-raw' in TOKS and 'BanglaBERT' not in TOKS:
    warn('BanglaBERT main-table entry is the UN-normalized variant because the normalizer is unavailable')
print('Main-table tokenizers:', MAIN_TOKS)

# %% [markdown]
# ## 6. Rulers per sentence: words, characters, graphemes, bytes, tokens

# %%
TEXTS = {'eng': EN, 'ben': BN}
PS = pd.DataFrame({'sent_idx': range(N), 'text_eng': EN, 'text_ben': BN})
for lang, texts in TEXTS.items():
    PS[f'words_{lang}'] = [len(s.split()) for s in texts]
    PS[f'chars_{lang}'] = [len(s) for s in texts]
    PS[f'graphemes_{lang}'] = [len(regex.findall(r'\X', s)) for s in texts]
    PS[f'bytes_{lang}'] = [len(s.encode('utf-8')) for s in texts]

IDS, INPUT = {}, {}
for name, T in TOKS.items():
    tok = T['tok']
    special = set(tok.all_special_ids) - {tok.unk_token_id}
    for lang, texts in TEXTS.items():
        inp = [T['pre'](s) for s in texts] if T['pre'] else list(texts)
        ids = tok(inp, add_special_tokens=False)['input_ids']
        hits = sum(i in special for x in ids for i in x)
        if hits:
            warn(f'{name}/{lang}: {hits} special-token ids appear despite add_special_tokens=False')
        IDS[(name, lang)], INPUT[(name, lang)] = ids, inp
        PS[f'tok_{name}_{lang}'] = [len(x) for x in ids]
    PS[f'ratio_{name}'] = PS[f'tok_{name}_ben'] / PS[f'tok_{name}_eng']

RULERS = {'words': 'words', 'characters': 'chars', 'graphemes': 'graphemes', 'bytes': 'bytes'}
RULERS.update({f'tokens: {n}': f'tok_{n}' for n in TOKS})
print(PS[[c for c in PS.columns if c.endswith(('_eng', '_ben')) and not c.startswith('text')]].sum().to_string())

# %% [markdown]
# ## 7. Paired bootstrap (seed 0, 10,000 resamples, same resample indices for every ruler)

# %%
B = 10_000
BOOT_IDX = np.random.default_rng(0).integers(0, N, size=(B, N), dtype=np.int32)

def ratio_stats(bn, en):
    bn = np.asarray(bn, float); en = np.asarray(en, float)
    boot = bn[BOOT_IDX].sum(1) / en[BOOT_IDX].sum(1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    with np.errstate(divide='ignore', invalid='ignore'):
        ps = bn / en
    ok = np.isfinite(ps)
    q1, med, q3 = np.percentile(ps[ok], [25, 50, 75])
    return dict(ratio=bn.sum() / en.sum(), ci_lo=lo, ci_hi=hi, median=med, q1=q1, q3=q3,
                cv=ps[ok].std(ddof=1) / ps[ok].mean(), n_undefined=int((~ok).sum()))

# %% [markdown]
# ## 8-9. Fertility and Bangla/English token ratio (main table)

# %%
rows = []
for name, T in TOKS.items():
    row = {'tokenizer': name, 'variant': T['variant'], 'repo': T['repo'], 'revision': T['revision'],
           'tokenizer_class': T['tokenizer_class'], 'vocab_size': T['vocab_size'],
           'vocab_len_incl_added': T['vocab_len_incl_added'],
           'native_window': T['native_window'] if T['native_window'] is not None else 'n/a',
           'role': T['role'], 'fallback_used': T['fallback_used'], 'in_main_table': name in MAIN_TOKS}
    for lang in ('eng', 'ben'):
        t = PS[f'tok_{name}_{lang}'].to_numpy(float)
        w = PS[f'words_{lang}'].to_numpy(float)
        c = PS[f'chars_{lang}'].to_numpy(float)
        row[f'tokens_total_{lang}'] = int(t.sum())
        row[f'fertility_corpus_{lang}'] = t.sum() / w.sum()
        row[f'fertility_mean_{lang}'] = (t / w).mean()
        row[f'tokens_per_char_{lang}'] = t.sum() / c.sum()
    s = ratio_stats(PS[f'tok_{name}_ben'], PS[f'tok_{name}_eng'])
    row.update({'ratio': s['ratio'], 'ratio_ci_lo': s['ci_lo'], 'ratio_ci_hi': s['ci_hi'],
                'ratio_median': s['median'], 'ratio_q1': s['q1'], 'ratio_q3': s['q3'], 'ratio_cv': s['cv']})
    row['ratio_note'] = 'not meaningful (Bangla-only tokenizer)' if name in NOT_MEANINGFUL else ''
    rows.append(row)
MAIN = pd.DataFrame(rows).set_index('tokenizer', drop=False)
MAIN.to_csv(os.path.join(OUT, 'results_main.csv'), index=False)
SHOW_MAIN = ['fertility_corpus_eng', 'fertility_corpus_ben', 'fertility_mean_eng', 'fertility_mean_ben',
             'tokens_per_char_eng', 'tokens_per_char_ben', 'ratio', 'ratio_ci_lo', 'ratio_ci_hi',
             'ratio_median', 'ratio_q1', 'ratio_q3', 'ratio_note']
print(MAIN[SHOW_MAIN].round(3).to_string())

# %% [markdown]
# ## 10. Effective context lost
# English-equivalent window = W / ratio; % lost = 1 - 1/ratio. "Sentences that fit" adds FLORES
# sentences in order and sums their per-sentence token counts (no separators), so it approximates
# a packed context. When the whole corpus fits, `corpus_exhausted` is True and the count is a floor.

# %%
WINDOWS = [512, 2048, 8192, 32768, 131072]
ec = []
for name, T in TOKS.items():
    r = MAIN.loc[name, 'ratio']
    cum = {lang: np.cumsum(PS[f'tok_{name}_{lang}'].to_numpy()) for lang in ('eng', 'ben')}
    wins = [(W, 'fixed') for W in WINDOWS]
    if T['native_window'] is None:
        ec.append({'tokenizer': name, 'window_type': 'native', 'W': np.nan,
                   'note': 'n/a: config has no max_position_embeddings'})
    else:
        wins.append((int(T['native_window']), 'native'))
    for W, wt in wins:
        d = {'tokenizer': name, 'window_type': wt, 'W': W, 'ratio': r,
             'english_equiv_window': W / r, 'pct_lost': 100 * (1 - 1 / r)}
        for lang in ('eng', 'ben'):
            d[f'n_fit_{lang}'] = int(np.searchsorted(cum[lang], W, side='right'))
            d[f'corpus_exhausted_{lang}'] = bool(cum[lang][-1] <= W)
        d['note'] = 'ratio not meaningful (Bangla-only tokenizer)' if name in NOT_MEANINGFUL else ''
        ec.append(d)
EC = pd.DataFrame(ec)
EC.to_csv(os.path.join(OUT, 'effective_context.csv'), index=False)
print(EC.round(1).to_string())

# %% [markdown]
# ## 11. Ruler comparison: which ruler is closest to 1, and most stable across sentences?

# %%
rc = []
for label, col in RULERS.items():
    s = ratio_stats(PS[f'{col}_ben'], PS[f'{col}_eng'])
    tname = label.split(': ', 1)[1] if label.startswith('tokens: ') else None
    rc.append({'ruler': label, 'kind': 'tokens' if tname else 'text unit', **s,
               'abs_dev_from_1': abs(s['ratio'] - 1),
               'note': 'not meaningful (Bangla-only tokenizer)' if tname in NOT_MEANINGFUL else ''})
RC = pd.DataFrame(rc).sort_values('abs_dev_from_1').reset_index(drop=True)
RC.to_csv(os.path.join(OUT, 'ruler_comparison.csv'), index=False)
print(RC.round(3).to_string())

# %% [markdown]
# ## 12. Diagnostics: UNK, partial characters, Bengali vocab, round trip, example pieces

# %%
BN_CHAR = regex.compile(r'[ঀ-৿]')
_vocab_cache = {}
def decode_each_id(tok):
    """Decode every vocab id on its own. Byte-level BPE vocab strings are byte-mapped,
    so counting Bengali characters on raw vocab strings would miss them."""
    key = id(tok)
    if key not in _vocab_cache:
        V = len(tok)
        if tok.is_fast:
            dec = tok.backend_tokenizer.decode_batch([[i] for i in range(V)], skip_special_tokens=False)
        else:
            dec = [tok.decode([i]) for i in range(V)]
        _vocab_cache[key] = dec
    return _vocab_cache[key]

def is_byte_level(tok):
    try:
        return type(tok.backend_tokenizer.decoder).__name__ == 'ByteLevel'
    except Exception:
        return False

DIAG = {'data': {k: DATA_INFO[k] for k in ('zw_counts_ben_before_nfc', 'zw_counts_ben_after_nfc',
                                            'sentences_changed_by_nfc')},
        'u+fffd_in_source_text': fffd_in_source, 'tokenizers': {}}
EX_ROWS = []
for name, T in TOKS.items():
    tok = T['tok']
    dec = decode_each_id(tok)
    n_bn_vocab = sum(1 for s in dec if BN_CHAR.search(s))
    d = {'repo': T['repo'], 'revision': T['revision'], 'variant': T['variant'],
         'vocab_entries_with_bengali_char': n_bn_vocab, 'vocab_entries_decoded': len(dec),
         'has_unk_token': tok.unk_token_id is not None, 'byte_level_bpe': is_byte_level(tok),
         'mirror_check': T['mirror_check'], 'per_lang': {}}
    for lang in ('eng', 'ben'):
        ids = IDS[(name, lang)]
        flat = [i for x in ids for i in x]
        n = len(flat)
        unk = tok.unk_token_id
        rt = [tok.decode(x, clean_up_tokenization_spaces=False) == s for x, s in zip(ids, INPUT[(name, lang)])]
        d['per_lang'][lang] = {
            'n_tokens': n,
            'unk_rate': (sum(i == unk for i in flat) / n) if unk is not None else 0.0,
            'partial_char_rate': sum('�' in dec[i] for i in flat) / n,
            'roundtrip_exact_rate': float(np.mean(rt)),
        }
        for i in EX_IDX:
            x = ids[i]
            raw_pieces = tok.convert_ids_to_tokens(x)
            dec_pieces = [dec[j] for j in x]
            EX_ROWS.append({'example_idx': i, 'lang': lang, 'tokenizer': name, 'n_tokens': len(x),
                            'input_text': INPUT[(name, lang)][i],
                            'pieces_decoded': ' | '.join(dec_pieces), 'pieces_raw': ' | '.join(raw_pieces),
                            'pieces_display': ' | '.join(dec_pieces if d['byte_level_bpe'] else raw_pieces)})
    DIAG['tokenizers'][name] = d

EX = pd.DataFrame(EX_ROWS)
EX.to_csv(os.path.join(OUT, 'example_tokenizations.csv'), index=False)
with open(os.path.join(OUT, 'diagnostics.json'), 'w', encoding='utf-8') as f:
    json.dump(DIAG, f, ensure_ascii=False, indent=2, default=str)

DIAG_TBL = pd.DataFrame([{'tokenizer': n, 'lang': l, **v, 'bn_vocab': d['vocab_entries_with_bengali_char']}
                         for n, d in DIAG['tokenizers'].items() for l, v in d['per_lang'].items()])
print(DIAG_TBL.round(4).to_string(index=False))
for i in EX_IDX:
    print(f'\n--- example {i} ---')
    for _, r in EX[EX.example_idx == i].iterrows():
        print(f"[{r.tokenizer}/{r.lang}] n={r.n_tokens}: {r.pieces_display}")

# %% [markdown]
# ## 13. Figures (English-only text; 150 dpi)

# %%
SURFACE, INK, INK2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'
C_EN, C_BN, C_NM = '#2a78d6', '#eb6834', '#a3a29c'
CAT = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#4a3aa7']
TOK_COLOR = {n: CAT[i % len(CAT)] for i, n in enumerate(TOKS)}   # fixed order, follows the entity
plt.rcParams.update({'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'axes.edgecolor': INK2,
                     'axes.labelcolor': INK, 'text.color': INK, 'xtick.color': INK2, 'ytick.color': INK2,
                     'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.8, 'axes.axisbelow': True,
                     'axes.spines.top': False, 'axes.spines.right': False, 'font.size': 10,
                     'savefig.dpi': 150, 'savefig.bbox': 'tight'})
def label_of(n):
    return n + ('*' if n in NOT_MEANINGFUL else '')
NM_FOOT = '* Bangla-only tokenizer: its Bangla/English ratio is not meaningful.'
def save(fig, fname):
    fig.savefig(os.path.join(FIG, fname)); plt.show(); plt.close(fig)

# 1. Fertility
names = MAIN_TOKS
x = np.arange(len(names)); w = 0.38
fig, ax = plt.subplots(figsize=(8, 4.2))
for k, (lang, col, lab) in enumerate([('eng', C_EN, 'English'), ('ben', C_BN, 'Bangla')]):
    vals = [MAIN.loc[n, f'fertility_corpus_{lang}'] for n in names]
    bars = ax.bar(x + (k - 0.5) * w, vals, w * 0.94, color=col, label=lab)
    ax.bar_label(bars, fmt='%.2f', fontsize=8, color=INK2, padding=2)
ax.set_xticks(x, [label_of(n) for n in names]); ax.set_ylabel('Tokens per whitespace word (corpus-level)')
ax.set_title('Fertility: tokens per word, FLORES-200 devtest', loc='left'); ax.legend(frameon=False)
ax.grid(axis='x', visible=False)
fig.text(0.01, -0.02, NM_FOOT, fontsize=8, color=INK2)
save(fig, 'fertility.png')

# 2. Token ratio with 95% CI
fig, ax = plt.subplots(figsize=(8, 4.2))
for i, n in enumerate(names):
    r = MAIN.loc[n]
    nm = n in NOT_MEANINGFUL
    ax.bar(i, r.ratio, 0.6, color=C_NM if nm else TOK_COLOR[n], hatch='//' if nm else None,
           edgecolor=SURFACE if not nm else INK2, linewidth=0.5)
    ax.errorbar(i, r.ratio, yerr=[[r.ratio - r.ratio_ci_lo], [r.ratio_ci_hi - r.ratio]], color=INK, capsize=4, lw=1.2)
    ax.text(i, r.ratio_ci_hi, ('not meaningful\n' if nm else '') + f'{r.ratio:.2f}', ha='center', va='bottom',
            fontsize=8, color=INK2)
ax.axhline(1, ls='--', color=INK, lw=1); ax.text(-0.45, 1, 'parity = 1', va='bottom', ha='left', fontsize=8, color=INK2)
ax.set_xticks(range(len(names)), [label_of(n) for n in names])
ax.set_ylabel('Bangla tokens / English tokens'); ax.grid(axis='x', visible=False)
ax.set_title('Bangla/English token ratio (corpus-level, 95% bootstrap CI)', loc='left')
ax.set_ylim(0, max(MAIN.loc[names, 'ratio_ci_hi']) * 1.25)
fig.text(0.01, -0.02, NM_FOOT, fontsize=8, color=INK2)
save(fig, 'token_ratio.png')

# 3. Ruler comparison
fig, ax = plt.subplots(figsize=(8, 0.42 * len(RC) + 1.2))
for i, r in RC.iloc[::-1].reset_index(drop=True).iterrows():
    nm = bool(r.note)
    col = C_NM if nm else ('#1baf7a' if r.kind == 'text unit' else '#4a3aa7')
    ax.barh(i, r.ratio, 0.6, color=col, hatch='//' if nm else None, edgecolor=INK2 if nm else SURFACE, linewidth=0.5)
    ax.errorbar(r.ratio, i, xerr=[[r.ratio - r.ci_lo], [r.ci_hi - r.ratio]], color=INK, capsize=3, lw=1)
    ax.text(max(r.ci_hi, 1.02), i, f'  {r.ratio:.2f}  (CV {r.cv:.2f})' + ('  not meaningful' if nm else ''),
            va='center', fontsize=8, color=INK2)
ax.set_yticks(range(len(RC)), [l + ('*' if l.split(': ')[-1] in NOT_MEANINGFUL else '') for l in RC.ruler[::-1]])
ax.axvline(1, ls='--', color=INK, lw=1)
ax.set_xlim(0, RC.ci_hi.max() * 1.45); ax.grid(axis='y', visible=False)
ax.set_xlabel('Bangla / English (corpus-level, 95% bootstrap CI)')
ax.set_title('Ruler comparison: green = text units, violet = tokens; sorted by |ratio - 1|', loc='left')
fig.text(0.01, -0.02, NM_FOOT + ' CV = coefficient of variation of per-sentence ratios.', fontsize=8, color=INK2)
save(fig, 'ruler_comparison.png')

# 4. Distribution of per-sentence ratios
fig, ax = plt.subplots(figsize=(8, 4.2))
data = [PS[f'ratio_{n}'].replace([np.inf, -np.inf], np.nan).dropna().to_numpy() for n in names]
parts = ax.violinplot(data, positions=range(len(names)), widths=0.8, showmedians=True, showextrema=False)
for body, n in zip(parts['bodies'], names):
    body.set_facecolor(C_NM if n in NOT_MEANINGFUL else TOK_COLOR[n]); body.set_alpha(0.75)
    body.set_edgecolor(SURFACE); body.set_linewidth(2)
    if n in NOT_MEANINGFUL: body.set_hatch('//')
parts['cmedians'].set_color(INK)
ax.axhline(1, ls='--', color=INK, lw=1)
ax.set_xticks(range(len(names)), [label_of(n) for n in names]); ax.grid(axis='x', visible=False)
ax.set_ylabel('Per-sentence Bangla/English token ratio')
ax.set_title('Distribution of per-sentence token ratios (line = median)', loc='left')
fig.text(0.01, -0.02, NM_FOOT, fontsize=8, color=INK2)
save(fig, 'ratio_distribution.png')

# 5. Effective context
fig, ax = plt.subplots(figsize=(8, 4.6))
Ws = np.array(WINDOWS, float)
ax.plot(Ws, Ws, ls='--', color=INK2, lw=1, label='parity (no loss)')
for n in names:
    sub = EC[(EC.tokenizer == n) & (EC.window_type == 'fixed')]
    nm = n in NOT_MEANINGFUL
    ax.plot(sub.W, sub.english_equiv_window, marker='o', ms=5, lw=2, ls=':' if nm else '-',
            color=C_NM if nm else TOK_COLOR[n], label=label_of(n) + (' (not meaningful)' if nm else ''))
    nat = EC[(EC.tokenizer == n) & (EC.window_type == 'native') & EC.W.notna()]
    if len(nat):
        ax.plot(nat.W, nat.english_equiv_window, marker='*', ms=13, ls='none',
                color=C_NM if nm else TOK_COLOR[n], markeredgecolor=SURFACE, markeredgewidth=1.5)
ax.set_xscale('log', base=2); ax.set_yscale('log', base=2)
_kfmt = matplotlib.ticker.FuncFormatter(lambda v, _: f'{v / 1024:g}k' if v >= 1024 else f'{v:g}')
ax.set_xticks(Ws); ax.xaxis.set_major_formatter(_kfmt); ax.yaxis.set_major_formatter(_kfmt)
ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
ax.set_xlabel('Context window W (tokens)'); ax.set_ylabel('English-equivalent window = W / ratio')
ax.set_title('Effective context for Bangla (star = model native window)', loc='left')
ax.legend(frameon=False, fontsize=8)
save(fig, 'effective_context.png')
print('Figures written:', sorted(os.listdir(FIG)))

# %% [markdown]
# ## 14. Pre-registered hypotheses: PASS / FAIL computed from the data

# %%
H1_THRESHOLD, DECISION_THRESHOLD = 1.5, 1.3
HYP = {}
present_llm = [n for n in LLM_NAMES if n in MAIN.index]
missing_llm = [n for n in LLM_NAMES if n not in MAIN.index]
llm_txt = ', '.join(f"{n}={MAIN.loc[n,'ratio']:.3f} [{MAIN.loc[n,'ratio_ci_lo']:.3f}, {MAIN.loc[n,'ratio_ci_hi']:.3f}]"
                    for n in present_llm)
if missing_llm:
    HYP['H1'] = ('INCONCLUSIVE', f'missing tokenizers {missing_llm}; loaded: {llm_txt}')
else:
    ok = all(MAIN.loc[n, 'ratio'] > H1_THRESHOLD for n in LLM_NAMES)
    ci_ok = all(MAIN.loc[n, 'ratio_ci_lo'] > H1_THRESHOLD for n in LLM_NAMES)
    HYP['H1'] = ('PASS' if ok else 'FAIL', f'{llm_txt}; threshold {H1_THRESHOLD}; all CI lower bounds above threshold: {ci_ok}')

fert_bn = MAIN.loc[MAIN_TOKS, 'fertility_corpus_ben'].sort_values()
bb = 'BanglaBERT' if 'BanglaBERT' in MAIN_TOKS else ('BanglaBERT-raw' if 'BanglaBERT-raw' in MAIN_TOKS else None)
fert_txt = ', '.join(f'{n}={v:.3f}' for n, v in fert_bn.items())
if bb is None:
    HYP['H2'] = ('INCONCLUSIVE', f'BanglaBERT not loaded; Bangla fertility: {fert_txt}')
else:
    note = '' if len(MAIN_TOKS) == 5 else f' (only {len(MAIN_TOKS)} of 5 tokenizers loaded)'
    HYP['H2'] = ('PASS' if fert_bn.index[0] == bb else 'FAIL', f'lowest = {fert_bn.index[0]}; Bangla fertility: {fert_txt}{note}')

rc_idx = RC.set_index('ruler')
text_best = min(['characters', 'graphemes'], key=lambda r: rc_idx.loc[r, 'abs_dev_from_1'])
tok_cands = [f'tokens: {n}' for n in MULTI_NAMES if f'tokens: {n}' in rc_idx.index]
if not tok_cands:
    HYP['H3'] = ('INCONCLUSIVE', 'no multilingual tokenizer loaded')
else:
    tok_best = min(tok_cands, key=lambda r: rc_idx.loc[r, 'abs_dev_from_1'])
    ok = rc_idx.loc[text_best, 'abs_dev_from_1'] < rc_idx.loc[tok_best, 'abs_dev_from_1']
    HYP['H3'] = ('PASS' if ok else 'FAIL',
                 f"best text ruler {text_best}={rc_idx.loc[text_best,'ratio']:.3f}; "
                 f"best multilingual token ruler {tok_best}={rc_idx.loc[tok_best,'ratio']:.3f}")

if missing_llm:
    DECISION = ('INCONCLUSIVE', f'cannot evaluate: missing {missing_llm}')
elif all(MAIN.loc[n, 'ratio'] < DECISION_THRESHOLD for n in LLM_NAMES):
    DECISION = ('TRIGGERED', f'All three modern LLM tokenizers have Bangla/English ratios below {DECISION_THRESHOLD} '
                f'({llm_txt}). The position-encoding motivation is WEAK for current models.')
else:
    DECISION = ('NOT TRIGGERED', f'at least one modern LLM has a ratio >= {DECISION_THRESHOLD} ({llm_txt})')
for k, (v, d) in HYP.items():
    print(f'{k}: {v} -- {d}')
print(f'Decision rule: {DECISION[0]} -- {DECISION[1]}')

# %% [markdown]
# ## 15. Write per_sentence.csv, summary.md, run_info.json

# %%
PS.to_csv(os.path.join(OUT, 'per_sentence.csv'), index=False)

def _ver(pkg):
    try:
        return importlib.metadata.version(pkg)
    except Exception:
        return None
import importlib.metadata
RUN_INFO = {
    'date_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
    'python': sys.version.split()[0], 'platform': platform.platform(), 'on_kaggle': ON_KAGGLE,
    'libraries': {p: _ver(p) for p in ('transformers', 'tokenizers', 'huggingface_hub', 'sentencepiece', 'protobuf',
                                       'regex', 'numpy', 'pandas', 'matplotlib', 'normalizer')},
    'data': {k: v for k, v in DATA_INFO.items()}, 'n_sentences_used': N, 'smoke_test_n': SMOKE_N or None,
    'example_indices': EX_IDX, 'bootstrap': {'resamples': B, 'seed': 0, 'paired_across_rulers': True},
    'tokenizers': {n: {k: T[k] for k in ('repo', 'revision', 'tokenizer_class', 'vocab_size', 'vocab_len_incl_added',
                                         'native_window', 'role', 'variant', 'fallback_used', 'mirror_check')}
                   for n, T in TOKS.items()},
    'skipped_tokenizers': SKIPPED, 'fallbacks': FALLBACKS, 'warnings': WARNINGS,
    'hypotheses': {k: {'verdict': v, 'detail': d} for k, (v, d) in HYP.items()},
    'decision_rule': {'verdict': DECISION[0], 'detail': DECISION[1]},
}

def md_table(df, fmt='{:.3f}'):
    cols = list(df.columns)
    cell = lambda v: fmt.format(v) if isinstance(v, (float, np.floating)) and np.isfinite(v) else ('' if v is None else str(v))
    lines = ['| ' + ' | '.join(map(str, cols)) + ' |', '|' + '---|' * len(cols)]
    lines += ['| ' + ' | '.join(cell(v) for v in r) + ' |' for r in df.itertuples(index=False)]
    return '\n'.join(lines)

EC_PIVOT = (EC[EC.window_type == 'fixed'].pivot(index='tokenizer', columns='W', values='english_equiv_window')
            .reindex(list(TOKS)).round(0).astype('Int64'))
EC_PIVOT.columns = [f'W={int(c)}' for c in EC_PIVOT.columns]
nat = EC[EC.window_type == 'native'].set_index('tokenizer')
EC_PIVOT['native W'] = [('n/a' if pd.isna(nat.loc[n, 'W']) else int(nat.loc[n, 'W'])) for n in EC_PIVOT.index]
EC_PIVOT['native eng-equiv'] = [('n/a' if pd.isna(nat.loc[n, 'W']) else int(round(nat.loc[n, 'english_equiv_window'])))
                                for n in EC_PIVOT.index]
EC_PIVOT['% lost'] = [round(100 * (1 - 1 / MAIN.loc[n, 'ratio']), 1) for n in EC_PIVOT.index]
EC_PIVOT['note'] = ['not meaningful' if n in NOT_MEANINGFUL else '' for n in EC_PIVOT.index]

def _fit(r):
    return (f"{int(r.n_fit_eng)}{'+' if r.corpus_exhausted_eng else ''}/"
            f"{int(r.n_fit_ben)}{'+' if r.corpus_exhausted_ben else ''}")
FIT = EC[EC.W.notna()].assign(fit=lambda d: d.apply(_fit, axis=1),
                              col=lambda d: d.apply(lambda r: 'native' if r.window_type == 'native' else f'W={int(r.W)}', axis=1))
FIT_PIVOT = FIT.pivot(index='tokenizer', columns='col', values='fit').reindex(list(TOKS))
FIT_PIVOT = FIT_PIVOT[[c for c in [f'W={w}' for w in WINDOWS] + ['native'] if c in FIT_PIVOT.columns]].fillna('n/a')

MAIN_MD = MAIN.loc[list(TOKS), ['tokenizer'] + SHOW_MAIN]
RC_MD = RC[['ruler', 'ratio', 'ci_lo', 'ci_hi', 'cv', 'median', 'abs_dev_from_1', 'note']]
summary = [
    '# How many tokens does Bangla cost across LLMs? -- summary', '',
    f"Data: {DATA_INFO['version']}; N = {N} sentence pairs" + (f' (SMOKE TEST, first {SMOKE_N})' if SMOKE_N else ''),
    f"Run: {RUN_INFO['date_utc']}; transformers {transformers.__version__}", '',
    '## Pre-registered hypotheses', '',
    *[f'- **{k}: {v}** -- {d}' for k, (v, d) in HYP.items()],
    f'- **Decision rule: {DECISION[0]}** -- {DECISION[1]}', '',
    '## Main table (corpus-level unless noted)', '', md_table(MAIN_MD), '',
    '## Ruler comparison (Bangla/English, sorted by |ratio - 1|)', '', md_table(RC_MD), '',
    '## Effective context: English-equivalent window (W / ratio)', '', md_table(EC_PIVOT.reset_index(), '{:.0f}'), '',
    '## Sentences that fit (English/Bangla; "+" = whole corpus fits)', '', md_table(FIT_PIVOT.reset_index()), '',
    '## Diagnostics', '', md_table(DIAG_TBL, '{:.4f}'), '',
    '## Skipped tokenizers', '', *([f'- {s}' for s in SKIPPED] or ['- none']), '',
    '## Fallbacks', '', *([f'- {s}' for s in FALLBACKS] or ['- none']), '',
    '## Warnings', '', *([f'- {s}' for s in WARNINGS] or ['- none']), '',
    'Notes: BanglaBERT is a Bangla-only tokenizer; its English numbers are reported but its ratio is not meaningful. '
    'Token counts exclude special tokens (add_special_tokens=False). "Sentences that fit" sums per-sentence counts. '
    'Example tokenizations: example_tokenizations.csv.',
]
with open(os.path.join(OUT, 'summary.md'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(summary) + '\n')
RUN_INFO['runtime_sec'] = round(time.time() - T0, 1)
with open(os.path.join(OUT, 'run_info.json'), 'w', encoding='utf-8') as f:
    json.dump(RUN_INFO, f, ensure_ascii=False, indent=2, default=str)
print('Wrote:', sorted(os.listdir(OUT)))

# %% [markdown]
# ## 16. Block to share

# %%
def _trunc(s, n=170):
    return s if len(s) <= n else s[:n - 3] + '...'
L = ['===== SHARE WITH CLAUDE: START =====']
L += [f"date {RUN_INFO['date_utc']} | python {RUN_INFO['python']} | runtime {RUN_INFO['runtime_sec']}s"
      + (f' | SMOKE TEST N={SMOKE_N}' if SMOKE_N else ''),
      'libs: ' + ', '.join(f'{k}={v}' for k, v in RUN_INFO['libraries'].items()),
      f"data: {DATA_INFO['version']} | N={N} | order check aligned={agree_aligned:.3f} shifted={agree_shifted:.3f}",
      f"ZWJ/ZWNJ (ben) before NFC {zw_before} after {zw_after} | NFC-changed sentences {nfc_changed}"]
for n, T in TOKS.items():
    L.append(f"tok {n}: {T['repo']}@{T['revision'][:12]} {T['tokenizer_class']} vocab={T['vocab_size']} "
             f"(+added {T['vocab_len_incl_added']}) native={T['native_window'] if T['native_window'] is not None else 'n/a'}"
             + (f" | {T['mirror_check']}" if T['mirror_check'] else ''))
L += ['', '## MAIN (fert = tokens/word, tpc = tokens/char; ratio = ben/eng corpus [95% CI]; per-sentence median [IQR])']
for n in TOKS:
    r = MAIN.loc[n]
    L.append(f"{n:15s} fert {r.fertility_corpus_eng:.3f}/{r.fertility_corpus_ben:.3f} mean {r.fertility_mean_eng:.3f}/"
             f"{r.fertility_mean_ben:.3f} tpc {r.tokens_per_char_eng:.3f}/{r.tokens_per_char_ben:.3f} "
             f"ratio {r.ratio:.3f} [{r.ratio_ci_lo:.3f},{r.ratio_ci_hi:.3f}] med {r.ratio_median:.3f} "
             f"[{r.ratio_q1:.3f},{r.ratio_q3:.3f}]" + (' NOT MEANINGFUL' if r.ratio_note else ''))
L += ['', '## RULERS (ben/eng corpus [95% CI], CV of per-sentence ratios), sorted by |ratio-1|']
for r in RC.itertuples():
    L.append(f"{r.ruler:26s} {r.ratio:.3f} [{r.ci_lo:.3f},{r.ci_hi:.3f}] CV {r.cv:.3f}" + (' NOT MEANINGFUL' if r.note else ''))
L += ['', '## EFFECTIVE CONTEXT: English-equivalent window = W/ratio']
L += EC_PIVOT.to_string().split('\n')
L += ['', '## SENTENCES THAT FIT eng/ben ("+" = corpus exhausted)']
L += FIT_PIVOT.to_string().split('\n')
L += ['', '## DIAGNOSTICS (unk, partial-char, roundtrip, #vocab entries with Bengali char)']
for r in DIAG_TBL.itertuples():
    L.append(f"{r.tokenizer:15s} {r.lang} unk={r.unk_rate:.4f} partial={r.partial_char_rate:.4f} "
             f"roundtrip={r.roundtrip_exact_rate:.3f} bn_vocab={r.bn_vocab}")
L += ['', '## HYPOTHESES']
L += [f'{k}: {v} -- {_trunc(d, 300)}' for k, (v, d) in HYP.items()]
L += [f'DECISION RULE: {DECISION[0]} -- {DECISION[1]}']
L += ['', f'## SKIPPED ({len(SKIPPED)}) / FALLBACKS ({len(FALLBACKS)}) / WARNINGS ({len(WARNINGS)})']
L += [_trunc('skipped: ' + s, 250) for s in SKIPPED] + [_trunc('fallback: ' + s, 250) for s in FALLBACKS]
L += [_trunc('warning: ' + s, 250) for s in WARNINGS[:12]]
if len(WARNINGS) > 12:
    L.append(f'... {len(WARNINGS) - 12} more warnings in run_info.json')
L += ['', '## EXAMPLES (byte-level BPE pieces shown decoded; others as raw pieces; " | " separates tokens)']
for i in EX_IDX:
    L.append(f'# example {i}: EN: {_trunc(EN[i], 150)}')
    L.append(f'#            BN: {_trunc(BN[i], 150)}')
    for r in EX[(EX.example_idx == i) & EX.tokenizer.isin(MAIN_TOKS)].itertuples():
        L.append(_trunc(f'{r.tokenizer}/{r.lang} n={r.n_tokens}: {r.pieces_display}'))
L.append('===== SHARE WITH CLAUDE: END =====')
print('\n'.join(L))
print(f'({len(L)} lines)')
