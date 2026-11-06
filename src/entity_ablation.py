"""Entity-level ablation of neuro-symbolic validation on manually annotated medication records (n2c2 2018 Track 2).

Pre-specified before the data were obtained (protocol frozen and hashed; see PROTOCOL below). The gold medication
annotations of n2c2 2018 Track 2 (Henry et al., 2020) supply the reference: each Drug entity and the Strength, Dosage,
Frequency, Route, Duration and Form entities linked to it by gold relations form one medication tuple. Recognition is
simulated by corrupting every entity value with an error model estimated from the executed Appendix M recognizer on
real RxHandBD prescription handwriting (and, as a sensitivity level, the Appendix K recognizer). Candidate regions and
medication grouping are provided from the gold annotation (the "provided regions" input condition of Section 3.10.7),
so that recognition noise and extraction are identical across conditions and only the acceptance layer changes.

Conditions (identical noisy candidates in all four):
  C0  no validation: every candidate tuple is accepted
  C1  terminology lookup: accept a tuple only if its normalized drug name is an exact development-lexicon entry
  C2  full validation: C1 plus the consistency rules (value-format rules for strength, dosage and duration; controlled
      vocabularies for frequency, route and form, built from development data)
  C3  full validation with lexicon-guided correction: each value is first replaced by its nearest development-lexicon
      or vocabulary entry when the normalized edit distance is <= delta and the best entry leads the second by >= margin;
      then C2 is applied
Endpoints per condition (official test split; unit = medication tuple and typed entity):
  accepted tuples, incorrect accepted tuples, accepted-tuple precision (Wilson 95%), tuple coverage, review rate,
  whole-test tuple recall; entity-level micro precision, recall and F1 with withheld tuples counted as false negatives;
  paired differences C3-C1, C2-C1, C1-C0 by 2,000 document-level bootstrap replicates (seed 42).
Development: the official training split only. 80% of its documents (seed 42) build the lexicon and vocabularies; the
remaining 20% tune delta and margin for C3 on whole-test tuple recall subject to accepted precision >= C1's.
"""
import collections, json, math, random, re, sys, unicodedata, hashlib
from pathlib import Path

SEED, B = 42, 2000
ATTR = ['Strength', 'Dosage', 'Frequency', 'Route', 'Duration', 'Form']
DELTAS = [0.0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
MARGINS = [0.0, 0.05, 0.1, 0.15]

# ---------------------------------------------------------------- normalization and edit distance
def norm(s):
    s = unicodedata.normalize('NFC', s or '')
    return re.sub(r'\s+', ' ', s).strip().casefold()

def lev(a, b):
    if a == b: return 0
    if len(a) < len(b): a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]

def align(r, h):
    """Levenshtein alignment -> list of (op, ref_char, hyp_char)."""
    n, m = len(r), len(h)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1): d[i][0] = i
    for j in range(m + 1): d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = min(d[i-1][j-1] + (r[i-1] != h[j-1]), d[i-1][j] + 1, d[i][j-1] + 1)
    ops, i, j = [], n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i][j] == d[i-1][j-1] + (r[i-1] != h[j-1]):
            ops.append(('H' if r[i-1] == h[j-1] else 'S', r[i-1], h[j-1])); i -= 1; j -= 1
        elif i > 0 and d[i][j] == d[i-1][j] + 1:
            ops.append(('D', r[i-1], '')); i -= 1
        else:
            ops.append(('I', '', h[j-1])); j -= 1
    return ops[::-1]

# ---------------------------------------------------------------- recognition error model (from executed runs)
class ErrorModel:
    """Two-stage model fitted to (reference, recognizer output) pairs: a word is read correctly with the observed
    exact-agreement probability; otherwise characters are corrupted at the conditional rates of the incorrect words,
    with substitutions drawn from the observed confusions and at least one error per corrupted word."""
    def __init__(self, pairs):
        pairs = [(norm(a), norm(b)) for a, b in pairs]
        self.p_exact = sum(a == b for a, b in pairs) / len(pairs)
        N = S = D = I = 0
        self.conf = collections.defaultdict(collections.Counter); self.ins = collections.Counter()
        for a, b in pairs:
            if a == b: continue
            N += len(a)
            for op, rc, hc in align(a, b):
                if op == 'S': S += 1; self.conf[rc][hc] += 1
                elif op == 'D': D += 1
                elif op == 'I': I += 1; self.ins[hc] += 1
        self.ps, self.pd, self.pi = S / N, D / N, I / N
        self.all_hyp = collections.Counter(c for k in self.conf.values() for c in k.elements()) or collections.Counter('a')
        self.pairs = len(pairs)

    def describe(self):
        return {'pairs': self.pairs, 'p_exact_word': self.p_exact, 'p_sub': self.ps, 'p_del': self.pd, 'p_ins': self.pi}

    def _pick(self, counter, rng):
        items = list(counter.items()); tot = sum(c for _, c in items); x = rng.random() * tot
        for k, c in items:
            x -= c
            if x <= 0: return k
        return items[-1][0]

    def corrupt_word(self, w, rng):
        if not w or rng.random() < self.p_exact: return w
        for _ in range(20):
            out = []
            for ch in w:
                x = rng.random()
                if x < self.pd: pass
                elif x < self.pd + self.ps:
                    out.append(self._pick(self.conf.get(ch.casefold()) or self.all_hyp, rng))
                else: out.append(ch)
                if rng.random() < self.pi: out.append(self._pick(self.ins or self.all_hyp, rng))
            o = ''.join(out)
            if norm(o) != norm(w): return o
        return w[:-1] if len(w) > 1 else w + 'l'

    def corrupt(self, text, rng):
        return ' '.join(self.corrupt_word(t, rng) for t in text.split(' '))

# ---------------------------------------------------------------- n2c2 2018 Track 2 (BRAT standoff) loading
def load_brat(folder):
    docs = {}
    for ann in sorted(Path(folder).rglob('*.ann')):
        ents, rels = {}, []
        for line in ann.read_text(encoding='utf-8', errors='replace').splitlines():
            if line.startswith('T'):
                tid, mid, text = line.split('\t', 2)
                typ = mid.split(' ', 1)[0]
                ents[tid] = {'type': typ, 'text': text.strip()}
            elif line.startswith('R'):
                rid, mid = line.split('\t')[:2]
                typ, a1, a2 = mid.split()
                rels.append((typ, a1.split(':')[1], a2.split(':')[1]))
        tuples = []
        drug_attrs = collections.defaultdict(list)
        for typ, a1, a2 in rels:
            attr, drug = (a1, a2) if ents.get(a2, {}).get('type') == 'Drug' else (a2, a1)
            if ents.get(drug, {}).get('type') == 'Drug' and ents.get(attr, {}).get('type') in ATTR:
                drug_attrs[drug].append(attr)
        for tid, e in ents.items():
            if e['type'] == 'Drug':
                fields = [('Drug', e['text'])] + sorted((ents[a]['type'], ents[a]['text']) for a in drug_attrs[tid])
                tuples.append(fields)
        docs[ann.stem] = tuples
    return docs

# ---------------------------------------------------------------- development resources
UNIT = r'(?:mg|mcg|g|gm|grams?|ml|cc|l|units?|u|iu|meq|mmol|%|puffs?|drops?|tabs?|tablets?|caps?|capsules?|patch)'
NUM = r'(?:\d+(?:[.,]\d+)?(?:\s*-\s*\d+(?:[.,]\d+)?)?|one|two|three|four|five|half|1/2)'
FORMAT = {
    'Strength': re.compile(r'^' + NUM + r'\s*' + UNIT + r'(?:\s*/\s*' + NUM + r'?\s*' + UNIT + r')?$'),
    'Dosage': re.compile(r'^' + NUM + r'(?:\s*' + UNIT + r')?(?:\s+\w+)?$'),
    'Duration': re.compile(r'^(?:for\s+|x\s*|times\s+)?(?:' + NUM + r'|a|an|the|next|several|few)\s*(?:more\s+)?'
                           r'(?:days?|d|weeks?|wks?|months?|doses?|hours?|hrs?)(?:\s+\w+)?$'),
}
VOCAB_TYPES = ['Frequency', 'Route', 'Form']

def build_resources(docs):
    lex = collections.Counter(); vocab = {t: collections.Counter() for t in VOCAB_TYPES}
    for tuples in docs.values():
        for t in tuples:
            for typ, v in t:
                if typ == 'Drug': lex[norm(v)] += 1
                elif typ in vocab: vocab[typ][norm(v)] += 1
    return {'lexicon': lex, 'vocab': vocab}

def nearest(value, entries, delta, margin):
    v = norm(value)
    if not entries or v in entries: return value, False
    best = sorted(((lev(v, e) / max(len(v), len(e), 1), -entries[e], e) for e in entries))[:2]
    d1 = best[0][0]; d2 = best[1][0] if len(best) > 1 else 1.0
    if d1 <= delta and d2 - d1 >= margin:
        return best[0][2], True
    return value, False

# ---------------------------------------------------------------- conditions
def field_ok(typ, v, res):
    n = norm(v)
    if typ == 'Drug': return n in res['lexicon']
    if typ in FORMAT: return bool(FORMAT[typ].match(n))
    if typ in res['vocab']: return n in res['vocab'][typ]
    return True

def decide(noisy, res, cond, delta=None, margin=None):
    """Return (accepted, output tuple). Output values are what an accepted record would contain."""
    out = list(noisy)
    if cond == 'C0': return True, out
    if cond == 'C3':
        fixed = []
        for typ, v in out:
            if typ == 'Drug': v, _ = nearest(v, res['lexicon'], delta, margin)
            elif typ in res['vocab']: v, _ = nearest(v, res['vocab'][typ], delta, margin)
            fixed.append((typ, v))
        out = fixed
    if not field_ok('Drug', out[0][1], res): return False, out
    if cond == 'C1': return True, out
    return all(field_ok(t, v, res) for t, v in out[1:]), out

# ---------------------------------------------------------------- evaluation
def tup_key(t): return sorted((typ, norm(v)) for typ, v in t)

def evaluate(items, cond, res, delta=None, margin=None):
    """items: list of (doc_id, gold_tuple, noisy_tuple). Returns per-doc stats."""
    per = collections.defaultdict(lambda: collections.Counter())
    for doc, gold, noisy in items:
        acc, out = decide(noisy, res, cond, delta, margin)
        c = per[doc]
        c['n'] += 1; c['ent_ref'] += len(gold)
        if acc:
            c['acc'] += 1
            ok = tup_key(out) == tup_key(gold)
            c['acc_ok'] += ok; c['acc_bad'] += (not ok)
            g = collections.Counter(tup_key(gold)); p = collections.Counter(tup_key(out))
            tp = sum((g & p).values()); c['ent_tp'] += tp; c['ent_pred'] += sum(p.values())
    return per

def metrics(per, docs=None):
    T = collections.Counter()
    for d in (docs if docs is not None else per):
        T.update(per[d])
    n = T['n'] or 1
    P = T['ent_tp'] / T['ent_pred'] if T['ent_pred'] else None
    R = T['ent_tp'] / T['ent_ref'] if T['ent_ref'] else None
    F = 2 * T['ent_tp'] / (T['ent_pred'] + T['ent_ref']) if (T['ent_pred'] + T['ent_ref']) else None
    return {'tuples': T['n'], 'accepted': T['acc'], 'incorrect_accepted': T['acc_bad'],
            'accepted_precision': T['acc_ok'] / T['acc'] if T['acc'] else None,
            'coverage': T['acc'] / n, 'review_rate': 1 - T['acc'] / n, 'tuple_recall': T['acc_ok'] / n,
            'incorrect_accepted_rate': T['acc_bad'] / n, 'entity_precision': P, 'entity_recall': R, 'entity_F1': F}

def wilson(k, n, z=1.959963984540054):
    if not n: return [None, None]
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [c - h, c + h]

def paired(perA, perB, key):
    docs = sorted(perA); rng = random.Random(SEED); diffs = []
    for _ in range(B):
        s = [docs[rng.randrange(len(docs))] for _ in docs]
        a, b = metrics(perA, s)[key], metrics(perB, s)[key]
        if a is not None and b is not None: diffs.append(a - b)
    diffs.sort()
    a, b = metrics(perA)[key], metrics(perB)[key]
    return {'diff': None if a is None or b is None else a - b,
            'ci95': [diffs[int(.025 * len(diffs))], diffs[int(.975 * len(diffs)) - 1]] if diffs else [None, None]}

def make_items(docs, model, seed):
    rng = random.Random(seed); items = []
    for d in sorted(docs):
        for t in docs[d]:
            noisy = [(typ, model.corrupt(v, rng)) for typ, v in t] if model else list(t)
            items.append((d, t, noisy))
    return items

def tune(items, res):
    base = metrics(evaluate(items, 'C1', res))['accepted_precision'] or 0
    best = None
    for dl in DELTAS:
        for mg in MARGINS:
            m = metrics(evaluate(items, 'C3', res, dl, mg))
            ok = (m['accepted_precision'] or 0) >= base
            key = (ok, m['tuple_recall'], -dl, -mg)
            if best is None or key > best[0]: best = (key, dl, mg, m)
    return {'delta': best[1], 'margin': best[2], 'tune_C1_precision': base, 'tune_C3': best[3]}

def run(train_dir, test_dir, pairs_by_level, out_path):
    train, test = load_brat(train_dir), load_brat(test_dir)
    ids = sorted(train); random.Random(SEED).shuffle(ids)
    cut = int(.8 * len(ids)); lex_docs = {d: train[d] for d in ids[:cut]}; tune_docs = {d: train[d] for d in ids[cut:]}
    res_tune = build_resources(lex_docs); res_test = build_resources(train)
    out = {'documents': {'train': len(train), 'test': len(test)},
           'tuples': {'train': sum(map(len, train.values())), 'test': sum(map(len, test.values()))},
           'lexicon_entries': len(res_test['lexicon']), 'levels': {}}
    for level, pairs in pairs_by_level.items():
        model = ErrorModel(pairs) if pairs else None
        sel = tune(make_items(tune_docs, model, SEED), res_tune)
        items = make_items(test, model, SEED + 1)
        per = {c: evaluate(items, c, res_test, sel['delta'], sel['margin']) for c in ('C0', 'C1', 'C2', 'C3')}
        M = {c: metrics(p) for c, p in per.items()}
        for c in M:
            k = round((M[c]['accepted_precision'] or 0) * M[c]['accepted'])
            M[c]['accepted_precision_wilson95'] = wilson(k, M[c]['accepted'])
        out['levels'][level] = {
            'error_model': model.describe() if model else 'none (clean text)', 'selection': sel, 'conditions': M,
            'paired': {f'{a}-{b}': {k: paired(per[a], per[b], k) for k in
                                    ('tuple_recall', 'incorrect_accepted_rate', 'accepted_precision', 'coverage', 'entity_F1')}
                       for a, b in (('C1', 'C0'), ('C2', 'C1'), ('C3', 'C1'), ('C3', 'C2'))}}
    json.dump(out, open(out_path, 'w'), indent=1)
    return out

if __name__ == '__main__':
    print(hashlib.sha256(open(__file__, 'rb').read()).hexdigest())
