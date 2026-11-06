"""Entity-level ablation of neuro-symbolic validation on PHEE (Sun et al., 2022), pre-specified before the test split
is scored. Reuses the recognition error model and evaluation of entity_ablation.py (n2c2 protocol), adapted to PHEE.

Reference: PHEE treatment arguments from MEDLINE case reports annotated with Drug and its Dosage, Route, Frequency and
Duration sub-arguments. A medication tuple = one Treatment argument with exactly one Drug and no Combination
sub-event (pre-specified), with its Dosage, Route, Freq and Duration values. Official splits: train builds the
lexicon and vocabularies, dev tunes C3 (delta, margin), test is scored once.
Rules (C2): drug in development lexicon; Route and Frequency in development vocabularies or matching a frequency
pattern; Dosage and Duration in development vocabularies or matching a quantity pattern.
"""
import collections, json, random, re, sys, hashlib
sys.path.insert(0, '/home/claude/n2c2')
import entity_ablation as EA
from entity_ablation import norm, nearest, wilson, paired, ErrorModel, SEED

MAP = {'Dosage': 'Dosage', 'Route': 'Route', 'Freq': 'Frequency', 'Duration': 'Duration'}
NUM = r'(?:\d+(?:[.,]\d+)?(?:\s*(?:-|to)\s*\d+(?:[.,]\d+)?)?|a|an|one|two|three|four|five|six|seven|eight|nine|ten|single)'
UNIT = r'(?:mg|mcg|µg|ug|g|gm|ml|l|units?|iu|u|meq|mmol|%|tablets?|capsules?|doses?|cycles?|courses?|injections?)'
PER = r'(?:\s*(?:/|per)\s*(?:kg|m2|m\(2\)|day|d|week|wk|h|hr|hour|dose|month|body weight)(?:\s*/\s*(?:day|d|week|h))?)*'
PAT = {
    'Dosage': re.compile(r'^' + NUM + r'\s*' + UNIT + PER + r'$'),
    'Duration': re.compile(r'^(?:for\s+)?' + NUM + r'[\s-]*(?:days?|weeks?|months?|years?|hours?|h|cycles?|courses?)$'),
    'Frequency': re.compile(r'^(?:' + NUM + r'\s*times?\s*(?:/|a|per|daily|weekly)?\s*\w*|every\s+' + NUM +
                            r'\s*\w+|q\d+h|(?:once|twice|thrice)\s*\w*|daily|weekly|monthly)$'),
}

def load(path):
    docs = collections.defaultdict(list); skipped = collections.Counter()
    for line in open(path):
        d = json.loads(line)
        for a in d['annotations']:
            for e in a['events']:
                t = e.get('Treatment')
                if not t: continue
                drugs = [' '.join(x) for x in t['Drug']['text']] if 'Drug' in t else []
                if len(drugs) != 1 or 'Combination' in t:
                    skipped['not_single_drug'] += 1; continue
                fields = [('Drug', drugs[0])]
                for k, typ in MAP.items():
                    if k in t:
                        for x in t[k]['text']: fields.append((typ, ' '.join(x)))
                docs[d['id'].split('_')[0]].append(fields)
    return dict(docs), skipped

RX = [n for n in (norm(x['name']) for x in json.load(open('/home/claude/phee/rxnorm_allconcepts.json'))) if n]

def resources(docs):
    """Terminology = RxNorm ingredient, precise-ingredient, multi-ingredient and brand names (RxNav allconcepts,
    retrieved Oct 29, 2026) plus drug names in the development documents; vocabularies from development documents."""
    lex = collections.Counter({n: 1 for n in RX}); voc = {t: collections.Counter() for t in MAP.values()}
    for tuples in docs.values():
        for t in tuples:
            for typ, v in t:
                (lex if typ == 'Drug' else voc[typ])[norm(v)] += 1
    return {'lexicon': lex, 'vocab': voc}

_NC = {}
def nearest(value, entries, delta, margin):
    """Cached nearest-entry search (length-prefiltered); same rule as entity_ablation.nearest."""
    v = norm(value)
    if not entries or v in entries: return value, False
    key = (v, id(entries))
    if key not in _NC:
        best = []
        for e in entries:
            if abs(len(e) - len(v)) > max(3, len(v) // 2): continue
            best.append((EA.lev(v, e) / max(len(v), len(e), 1), -entries[e], e))
        best.sort(); _NC[key] = best[:2]
    best = _NC[key]
    if not best: return value, False
    d1 = best[0][0]; d2 = best[1][0] if len(best) > 1 else 1.0
    return (best[0][2], True) if d1 <= delta and d2 - d1 >= margin else (value, False)

def field_ok(typ, v, res):
    n = norm(v)
    if typ == 'Drug': return n in res['lexicon']
    if n in res['vocab'].get(typ, ()): return True
    return bool(PAT[typ].match(n)) if typ in PAT else False

def decide(noisy, res, cond, delta=None, margin=None):
    out = list(noisy)
    if cond == 'C0': return True, out
    if cond == 'C3':
        out = [(typ, nearest(v, res['lexicon'] if typ == 'Drug' else (res['vocab'][typ] if typ in ('Route', 'Frequency')
                                 else {}), delta, margin)[0]) for typ, v in out]
    if not field_ok('Drug', out[0][1], res): return False, out
    if cond == 'C1': return True, out
    return all(field_ok(t, v, res) for t, v in out[1:]), out

EA.decide = decide   # evaluation in EA calls decide()

def run(out_path, pairs_by_level):
    train, s1 = load('/home/claude/phee/data/train.json'); dev, s2 = load('/home/claude/phee/data/dev.json')
    test, s3 = load('/home/claude/phee/data/test.json')
    res_tr = resources(train); res_test = resources({**train, **dev})
    out = {'tuples': {'train': sum(map(len, train.values())), 'dev': sum(map(len, dev.values())),
                      'test': sum(map(len, test.values()))},
           'test_tuples_with_attributes': sum(len(t) > 1 for v in test.values() for t in v),
           'skipped_not_single_drug': {'train': s1['not_single_drug'], 'dev': s2['not_single_drug'],
                                       'test': s3['not_single_drug']},
           'lexicon_entries': len(res_test['lexicon']), 'levels': {}}
    for level, pairs in pairs_by_level.items():
        model = ErrorModel(pairs) if pairs else None
        sel = EA.tune(EA.make_items(dev, model, SEED), res_tr)
        items = EA.make_items(test, model, SEED + 1)
        per = {c: EA.evaluate(items, c, res_test, sel['delta'], sel['margin']) for c in ('C0', 'C1', 'C2', 'C3')}
        M = {c: EA.metrics(p) for c, p in per.items()}
        for c in M:
            k = round((M[c]['accepted_precision'] or 0) * M[c]['accepted'])
            M[c]['accepted_precision_wilson95'] = wilson(k, M[c]['accepted'])
        out['levels'][level] = {'error_model': model.describe() if model else 'none (clean text)', 'selection': sel,
            'conditions': M,
            'paired': {f'{a}-{b}': {k: paired(per[a], per[b], k) for k in
                       ('tuple_recall', 'incorrect_accepted_rate', 'accepted_precision', 'coverage', 'entity_F1')}
                       for a, b in (('C1', 'C0'), ('C2', 'C1'), ('C3', 'C1'), ('C3', 'C2'))}}
    json.dump(out, open(out_path, 'w'), indent=1)
    return out

if __name__ == '__main__':
    print(hashlib.sha256(open(__file__, 'rb').read()).hexdigest())
