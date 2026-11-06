"""Word-level ablation of symbolic validation on RxHandBD (pre-specified before any data were opened).

Same saved TrOCR predictions on the 1,114 scorable test crops enter three conditions:
  C0  no gate                 accept every recognized output
  C1  exact-lexicon gate      accept only if the normalized output exactly matches the frozen lexicon
  C2  full symbolic gate      C1 plus the implemented consistency rules (Section 3.10.8):
                              numeric review flags, more than one lexicon term in one region
                              (ambiguous medication count), and lexicon ambiguity
The lexicon is frozen from development data only: the distinct labels of the RxHandBD training
partition (no test labels). A crop reading is correct when its normalized output equals the
normalized publisher label (primary scorer of Chapter 4). Outputs that are withheld are routed to
review; they are never corrected or replaced.
"""
import csv, json, math, random, re, sys, unicodedata
from pathlib import Path

SEED, B = 42, 2000

# ---------- primary scorer normalization (Section 4.3) ----------
def normalize(s):
    s = unicodedata.normalize('NFC', s or '')
    return re.sub(r'\s+', ' ', s).strip().casefold()

def scorable(ref):
    return not re.fullmatch(r'[?\s]+', ref or '')

# ---------- implemented numeric review rules (Appendix F, unchanged) ----------
ATOM = (r"[+\-−]?(?:\d+\s+\d+\s*/\s*\d+|\d+\s*/\s*\d+"
        r"|\d*[.,]\d+(?:[eE][+\-]?\d+)?|\d+[eE][+\-]?\d+"
        r"|\d+[¼½¾]|\d+|[¼½¾])")
NUMBER = r"(?:" + ATOM + r"\s*[–−-]\s*" + ATOM + r"|" + ATOM + r")"

def numeric_review_flags(value):
    flags = []
    if re.match(r'[+\-−]', value): flags.append('signed_quantity')
    if re.search(r'\d[eE][+\-]?\d', value): flags.append('scientific_quantity')
    if re.search(r'[\d¼½¾]\s*[–−-]\s*[+\-−]?\d', value): flags.append('quantity_range')
    if re.search(r'\d*,\d', value): flags.append('decimal_comma_quantity')
    if re.search(r'/\s*0(?:\D|$)', value): flags.append('zero_denominator')
    return flags

# ---------- input loading (flexible column names) ----------
def read_table(path):
    p = Path(path)
    if p.suffix.lower() == '.jsonl':
        return [json.loads(l) for l in p.read_text(encoding='utf-8').splitlines() if l.strip()]
    with open(p, encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def pick(row, names):
    low = {k.lower().strip(): k for k in row}
    for n in names:
        if n in low:
            return low[n]
    raise KeyError(f'None of {names} in columns {list(row)}')

def load_lexicon(train_csv):
    rows = read_table(train_csv)
    tc = pick(rows[0], ['text', 'label', 'transcription'])
    raw = {}
    for r in rows:
        if scorable(r[tc]):
            raw.setdefault(normalize(r[tc]), set()).add(r[tc].strip())
    return raw          # normalized form -> set of raw training spellings

def load_predictions(path, model=None):
    rows = read_table(path)
    r0 = rows[0]
    idc = pick(r0, ['record_id', 'image', 'images', 'image_id', 'id', 'filename'])
    refc = pick(r0, ['reference_text', 'reference', 'text', 'label', 'ref'])
    generic = ['prediction', 'pred', 'hypothesis', 'output', 'raw_prediction', 'raw_text']
    long_format = any(k.lower() in ('model', 'recognizer', 'system') for k in r0)
    if model and not long_format:
        pc = pick(r0, [model.lower(), model.lower() + '_prediction', 'prediction_' + model.lower(),
                       model.lower() + '_output'] + generic)
    elif model:
        pc = pick(r0, generic)
    else:
        pc = pick(r0, ['prediction', 'pred', 'hypothesis', 'output', 'trocr', 'trocr_prediction'])
    out = []
    for r in rows:
        mc = next((k for k in r if k.lower() in ('model', 'recognizer', 'system')), None)
        if model and mc and r[mc].strip().lower() != model.lower():
            continue
        out.append({'id': r[idc], 'ref': r[refc], 'hyp': r[pc] or ''})
    return out

# ---------- the three conditions ----------
def gate(hyp, lexicon, full):
    h = normalize(hyp)
    flags = []
    if h not in lexicon:
        flags.append('terminology_unresolved')
    if full:
        for m in re.finditer(NUMBER, hyp):
            flags += numeric_review_flags(m.group(0))
        tokens = h.split(' ') if h else []
        terms = {t for t in tokens if t in lexicon and len(tokens) > 1}
        if len(terms) > 1:
            flags.append('ambiguous_medication_count')
        if h in lexicon and len(lexicon[h]) > 1 and len({normalize(x) for x in lexicon[h]}) > 1:
            flags.append('terminology_ambiguous')
    return (len(flags) == 0), sorted(set(flags))

def run(preds, lexicon):
    recs = []
    for p in preds:
        correct = normalize(p['hyp']) == normalize(p['ref'])
        c1, f1 = gate(p['hyp'], lexicon, full=False)
        c2, f2 = gate(p['hyp'], lexicon, full=True)
        recs.append({'id': p['id'], 'ref': p['ref'], 'hyp': p['hyp'], 'correct': correct,
                     'ref_in_lexicon': normalize(p['ref']) in lexicon,
                     'C0': True, 'C1': c1, 'C2': c2, 'C1_flags': f1, 'C2_flags': f2})
    return recs

def metrics(recs, cond):
    n = len(recs)
    acc = [r for r in recs if r[cond]]
    ok = sum(r['correct'] for r in acc)
    wrong = len(acc) - ok
    withheld = [r for r in recs if not r[cond]]
    return {'n': n, 'accepted': len(acc), 'coverage': len(acc) / n,
            'accepted_correct': ok, 'incorrect_accepted': wrong,
            'accepted_precision': ok / len(acc) if acc else None,
            'incorrect_accepted_rate_of_all': wrong / n,
            'whole_test_recall': ok / n, 'review_rate': len(withheld) / n,
            'beneficial_withholds': sum(not r['correct'] for r in withheld),
            'harmful_withholds': sum(r['correct'] for r in withheld)}

def bootstrap(recs, a, b, key):
    rng = random.Random(SEED); n = len(recs); diffs = []
    for _ in range(B):
        s = [recs[rng.randrange(n)] for _ in range(n)]
        ma, mb = metrics(s, a)[key], metrics(s, b)[key]
        if ma is not None and mb is not None:
            diffs.append(ma - mb)
    diffs.sort()
    lo, hi = diffs[int(0.025 * len(diffs))], diffs[int(0.975 * len(diffs)) - 1]
    return {'diff': metrics(recs, a)[key] - metrics(recs, b)[key], 'ci95': [lo, hi], 'replicates': len(diffs)}

def wilson(k, n, z=1.959963984540054):
    if not n: return [None, None]
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [c - h, c + h]

def main(train_csv, pred_path, out_dir, model='trocr'):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    lex = load_lexicon(train_csv)
    preds = [p for p in load_predictions(pred_path, model) if scorable(p['ref'])]
    recs = run(preds, lex)
    res = {'model': model, 'lexicon_entries': len(lex), 'population': len(recs),
           'conditions': {c: metrics(recs, c) for c in ('C0', 'C1', 'C2')}}
    for c in res['conditions'].values():
        c['accepted_precision_wilson95'] = wilson(c['accepted_correct'], c['accepted'])
    res['paired'] = {f'{a}-{b}': {k: bootstrap(recs, a, b, k)
                                  for k in ('accepted_precision', 'incorrect_accepted_rate_of_all',
                                            'whole_test_recall', 'coverage')}
                     for a, b in (('C1', 'C0'), ('C2', 'C0'), ('C2', 'C1'))}
    res['by_reference_in_lexicon'] = {
        str(v): {c: metrics([r for r in recs if r['ref_in_lexicon'] == v], c) for c in ('C0', 'C1', 'C2')}
        for v in (True, False)}
    from collections import Counter
    res['C2_flag_counts'] = dict(Counter(f for r in recs for f in r['C2_flags']))
    json.dump(res, open(out / f'word_ablation_{model}.json', 'w'), indent=1)
    with open(out / f'word_ablation_{model}_records.jsonl', 'w', encoding='utf-8') as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    return res

if __name__ == '__main__':
    main(*sys.argv[1:])
