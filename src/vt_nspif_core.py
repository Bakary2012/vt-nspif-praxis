"""VT-NSPIF research workflow; no clinical or performance claims.

Revised companion to the Praxis protocol.
Only executed, auditable outputs may populate Chapter 4.
"""
from __future__ import annotations
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
import csv
import hashlib
import importlib.metadata
import json
import math
import platform
import random
import re
import subprocess
import time
import unicodedata
import urllib.parse
import urllib.request

ENTITY_TYPES = (
    'drug_name', 'strength', 'dose', 'frequency', 'route',
    'duration', 'conditional',
)

@dataclass(frozen=True)
class Config:
    seed: int = 42
    bootstrap_replicates: int = 2000
    checkpoint: str = 'microsoft/trocr-base-handwritten'
    checkpoint_revision: str = 'main'
    max_new_tokens: int = 128
    num_beams: int = 1
    fhir_version: str = '5.0.0'
    rule_version: str = 'vt-nspif-rules-2026-10-23'
    allow_network: bool = False
    required_fields: tuple = ('drug_name',)

CONFIG = Config()

def utc_now():
    return datetime.now(timezone.utc).isoformat()

def digest_bytes(content):
    return hashlib.sha256(content).hexdigest()

def digest_file(path):
    return digest_bytes(Path(path).read_bytes())

def canonical_json(obj):
    return json.dumps(obj, ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'), allow_nan=False)

def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False,
                    indent=2, allow_nan=False), encoding='utf-8')

def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as stream:
        for row in rows:
            stream.write(canonical_json(row) + '\n')

def read_jsonl(path):
    with Path(path).open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]

def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row})
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)

def normalize_text(value, strict=False):
    """NFC and whitespace; preserve scripts, decimals and punctuation."""
    value = unicodedata.normalize('NFC', str(value))
    value = ' '.join(value.split())
    return value if strict else value.casefold()

def ratio(numerator, denominator):
    return numerator / denominator if denominator else None

def safe_mean(values):
    defined = [v for v in values if v is not None]
    return sum(defined) / len(defined) if defined else None

def edit_counts(reference, prediction):
    """Deterministic minimum-distance alignment, S/D/I/H counts."""
    ref, pred = list(reference), list(prediction)
    matrix = [[0] * (len(pred) + 1)
              for _ in range(len(ref) + 1)]
    for i in range(len(ref) + 1):
        matrix[i][0] = i
    for j in range(len(pred) + 1):
        matrix[0][j] = j
    for i in range(1, len(ref) + 1):
        for j in range(1, len(pred) + 1):
            matrix[i][j] = min(
                matrix[i-1][j] + 1,
                matrix[i][j-1] + 1,
                matrix[i-1][j-1] + (ref[i-1] != pred[j-1]),
            )
    i, j = len(ref), len(pred)
    counts = Counter(S=0, D=0, I=0, H=0, N=len(ref))
    while i or j:
        if (i and j and matrix[i][j] == matrix[i-1][j-1]
                + (ref[i-1] != pred[j-1])):
            counts['H' if ref[i-1] == pred[j-1] else 'S'] += 1
            i, j = i-1, j-1
        elif i and matrix[i][j] == matrix[i-1][j] + 1:
            counts['D'] += 1
            i -= 1
        else:
            counts['I'] += 1
            j -= 1
    return dict(counts)

def text_scores(reference, prediction, strict=False):
    ref = normalize_text(reference, strict)
    pred = normalize_text(prediction, strict)
    chars = edit_counts(ref, pred)
    words = edit_counts(ref.split(), pred.split())
    return {
        'character_counts': chars, 'word_counts': words,
        'CER': ratio(chars['S']+chars['D']+chars['I'], chars['N']),
        'WER': ratio(words['S']+words['D']+words['I'], words['N']),
        'empty_reference': not bool(ref),
        'record_text_exact': ref == pred,
    }

def validate_manifest(rows, base_dir):
    """Fail on unverifiable labels, splits or provenance."""
    required = {
        'record_id', 'source', 'source_version', 'split',
        'authentic', 'prescription_group', 'language',
        'image_path', 'image_sha256', 'reference_text',
        'annotation_status', 'license', 'regions',
        'reference_entities', 'task',
    }
    seen, hashes = set(), {}
    group_splits = defaultdict(set)
    writer_splits = defaultdict(set)
    for row in rows:
        missing = required - row.keys()
        if missing:
            raise ValueError(f'Missing manifest fields: {sorted(missing)}')
        rid = row['record_id']
        if rid in seen:
            raise ValueError(f'Duplicate record_id: {rid}')
        seen.add(rid)
        if row['split'] not in {'train', 'val', 'test'}:
            raise ValueError(f'Invalid split: {rid}')
        if not isinstance(row['authentic'], bool):
            raise ValueError('authentic must be Boolean')
        if row['annotation_status'] != 'adjudicated':
            raise ValueError(f'Unaudited annotation: {rid}')
        if not normalize_text(row['reference_text']):
            raise ValueError(f'Empty primary transcription reference: {rid}')
        if not row['license'] or not row['prescription_group']:
            raise ValueError(f'Missing licensing/group evidence: {rid}')
        if row['task'] not in {'word', 'provided_regions'}:
            raise ValueError('Only word/provided-region tracks implemented')
        path = Path(base_dir) / row['image_path']
        if digest_file(path) != row['image_sha256']:
            raise ValueError(f'Image hash mismatch: {rid}')
        sha = row['image_sha256']
        if sha in hashes:
            raise ValueError(f'Duplicate image: {rid}, {hashes[sha]}')
        hashes[sha] = rid
        group_splits[row['prescription_group']].add(row['split'])
        if row.get('writer_id'):
            writer_splits[row['writer_id']].add(row['split'])
        groups = set()
        for region in row['regions']:
            for key in ('region_id', 'medication_group', 'box', 'order'):
                if key not in region:
                    raise ValueError(f'Missing region field: {key}')
            if region['region_id'] in groups:
                raise ValueError('Duplicate region identifier')
            groups.add(region['region_id'])
            if len(region['box']) != 4:
                raise ValueError('Region box requires x0,y0,x1,y1')
        for entity in row['reference_entities']:
            if entity['type'] not in ENTITY_TYPES:
                raise ValueError('Unsupported entity type')
            if not entity.get('value') or not entity.get('medication_group'):
                raise ValueError('Entity value/association is required')
            if entity['medication_group'] not in {
                    r['medication_group'] for r in row['regions']}:
                raise ValueError('Reference association lacks a provided region')
    if any(len(splits) > 1 for splits in group_splits.values()):
        raise ValueError('Prescription group crosses splits')
    if any(len(splits) > 1 for splits in writer_splits.values()):
        raise ValueError('Writer crosses splits; resolve before evaluation')
    return {'records': len(rows), 'unique_images': len(hashes),
            'writer_independence_supported': all(r.get('writer_id') for r in rows)}

def assign_group_splits(rows, seed=42):
    """Study-defined partition only; never overwrite benchmark splits."""
    if any(row.get('split') for row in rows):
        raise ValueError('Preserve existing splits; do not overwrite')
    if not all(row.get('writer_id') for row in rows):
        raise ValueError('Writer IDs required for automatic group partition')
    groups = sorted({row['writer_id'] for row in rows})
    if len(groups) < 10:
        raise ValueError('Too few writers for automatic 70/15/15 partition')
    random.Random(seed).shuffle(groups)
    a, b = int(.70*len(groups)), int(.85*len(groups))
    mapping = {g: 'train' if i < a else 'val' if i < b else 'test'
               for i, g in enumerate(groups)}
    result = deepcopy(rows)
    for row in result:
        row['split'] = mapping[row['writer_id']]
    return result

def load_lexicon(path):
    """Audited aliases, not synthetic-generator medicine names."""
    entries = json.loads(Path(path).read_text(encoding='utf-8'))
    aliases = {}
    for entry in entries:
        for alias in entry['aliases']:
            key = normalize_text(alias)
            if key in aliases and aliases[key]['rxcui'] != entry['rxcui']:
                raise ValueError(f'Ambiguous alias: {alias}')
            aliases[key] = entry
    return aliases

# Numeric alternatives are ordered to retain complete source expressions.
ATOM = (r"[+\-−]?(?:\d+\s+\d+\s*/\s*\d+|\d+\s*/\s*\d+"
        r"|\d*[.,]\d+(?:[eE][+\-]?\d+)?|\d+[eE][+\-]?\d+"
        r"|\d+[¼½¾]|\d+|[¼½¾])")
NUMBER = r"(?:" + ATOM + r"\s*[–−-]\s*" + ATOM + r"|" + ATOM + r")"
NUMERIC_START = r"(?<![\w.,/+–−-])"

def numeric_review_flags(value):
    """Preserve unusual expressions while requiring human interpretation."""
    flags = []
    if re.match(r'[+\-−]', value): flags.append('signed_quantity')
    if re.search(r'\d[eE][+\-]?\d', value): flags.append('scientific_quantity')
    if re.search(r'[\d¼½¾]\s*[–−-]\s*[+\-−]?\d', value):
        flags.append('quantity_range')
    if re.search(r'\d*,\d', value): flags.append('decimal_comma_quantity')
    if re.search(r'/\s*0(?:\D|$)', value): flags.append('zero_denominator')
    return flags
UNIT = r"(?:tablets?|capsules?|mL|mg|mcg|g|units?)"
PATTERNS = {
    'frequency': r'\b(?:once daily|twice daily|every \d+ hours|daily|bid|tid|qid|qhs)\b',
    'route': r'\b(?:PO|oral|by mouth|IV|IM|SC|subcutaneous)\b',
    'duration': r'\b(?:for\s+\d+\s+(?:days?|weeks?))\b',
    'conditional': r'\b(?:as needed|prn)(?:\s+for\s+[\w ]+)?\b',
}

def extract_group(text, group_id, aliases):
    """Conservative English fixtures; source values are never numerically rewritten."""
    entities, matches, occupied = [], [], []
    def emit(typ, start, end):
        entities.append({'type': typ, 'value': text[start:end],
            'span': [start, end], 'medication_group': group_id})
        if typ in {'strength', 'dose'}:
            entities[-1]['numeric_review_flags'] = numeric_review_flags(text[start:end])
    for alias in aliases:
        pattern = r'(?<!\w)' + re.escape(alias) + r'(?!\w)'
        for match in re.finditer(pattern, text, flags=re.I):
            matches.append((match.start(), match.end(), alias))
    selected = []
    for start, end, alias in sorted(matches, key=lambda x: -(x[1]-x[0])):
        if not any(start < b and end > a for a, b, _ in selected):
            selected.append((start, end, alias))
    for start, end, alias in sorted(selected):
        emit('drug_name', start, end)
        entities[-1]['alias_key'] = alias
    # A concentration is one product-strength span, not an administration dose.
    concentration = (NUMERIC_START + NUMBER + r'\s*(?:mg|mcg|g|units?)'
        + r'\s*/\s*(?:' + NUMBER + r'\s*)?(?:mL|g)\b')
    for m in re.finditer(concentration, text, re.I):
        emit('strength', *m.span()); occupied.append(m.span())
    dose_pattern = (r'\b(?:take|give|inject|administer|use)\s+('
        + NUMBER + r'\s*' + UNIT + r')\b(?!\s*/)')
    for m in re.finditer(dose_pattern, text, re.I):
        start, end = m.span(1)
        if not any(start < b and end > a for a,b in occupied):
            emit('dose', start, end); occupied.append((start,end))
    strength_pattern = NUMERIC_START + NUMBER + r'\s*(?:mg|mcg|g|units?)\b(?!\s*/)'
    for m in re.finditer(strength_pattern, text, re.I):
        start,end = m.span()
        if not any(start < b and end > a for a,b in occupied):
            emit('strength', start, end); occupied.append((start,end))
    for typ, pattern in PATTERNS.items():
        for m in re.finditer(pattern, text, re.I): emit(typ, *m.span())
    return sorted(entities, key=lambda e: (e['span'][0], e['type']))

class TerminologyClient:
    """Cache every response; no approximate automatic correction."""
    def __init__(self, cache_dir, allow_network=False):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.allow_network = allow_network

    def get(self, endpoint, params=None):
        params = params or {}
        url = 'https://rxnav.nlm.nih.gov/REST/' + endpoint
        if params:
            url += '?' + urllib.parse.urlencode(params)
        key = digest_bytes(url.encode())
        path = self.cache_dir / (key + '.json')
        if path.exists():
            return json.loads(path.read_text(encoding='utf-8'))
        if not self.allow_network:
            return {'status': 'unavailable_offline', 'url': url}
        try:
            with urllib.request.urlopen(url, timeout=15) as response:
                payload = json.load(response)
            result = {'status': 'ok', 'url': url,
                      'retrieved_at': utc_now(), 'payload': payload}
        except Exception as error:
            result = {'status': 'lookup_failure', 'url': url,
                      'retrieved_at': utc_now(), 'error': repr(error)}
        save_json(path, result)
        return result

    def lookup(self, term):
        exact = self.get('rxcui.json', {'name': term, 'search': 0})
        ids = exact.get('payload', {}).get('idGroup', {}).get('rxnormId', [])
        if len(ids) == 1:
            props = self.get(f'rxcui/{ids[0]}/properties.json')
            return {'status': 'exact', 'rxcui': ids[0],
                    'properties': props, 'trace': [exact]}
        if len(ids) > 1:
            return {'status': 'ambiguous', 'trace': [exact], 'ids': ids}
        approx = self.get('approximateTerm.json',
                          {'term': term, 'maxEntries': 5})
        candidates = approx.get('payload', {}).get(
            'approximateGroup', {}).get('candidate', [])
        return {'status': 'unresolved', 'candidates': candidates,
                'trace': [exact, approx]}

def validate_entities(entities, client, config=CONFIG):
    result = deepcopy(entities)
    grouped = defaultdict(list)
    for entity in result:
        grouped[entity['medication_group']].append(entity)
    traces, decisions = [], []
    for group, items in grouped.items():
        flags = sorted({flag for e in items for flag in e.get('numeric_review_flags', [])})
        drugs = [e for e in items if e['type'] == 'drug_name']
        if len(drugs) != 1:
            flags.append('ambiguous_medication_count')
        for typ in config.required_fields:
            if not any(e['type'] == typ for e in items):
                flags.append('missing_' + typ)
        for typ in ('strength', 'dose', 'frequency', 'route', 'duration'):
            if len([e for e in items if e['type'] == typ]) > 1:
                flags.append('multiple_' + typ)
        for entity in drugs:
            lookup = client.lookup(entity['value'])
            traces.append({'group': group, 'lookup': lookup})
            entity['terminology_status'] = lookup['status']
            if lookup['status'] == 'exact':
                entity['rxcui'] = lookup['rxcui']
                props = lookup.get('properties', {}).get('payload', {})
                props = props.get('properties', {})
                entity['canonical_display'] = props.get('name')
                entity['term_type'] = props.get('tty')
            else:
                flags.append('terminology_' + lookup['status'])
                entity['candidates'] = lookup.get('candidates', [])
        decisions.append({'medication_group': group,
            'review_required': bool(flags), 'flags': flags,
            'rule_version': config.rule_version})
    # Empty groups are separately retained by the runner.
    return {'entities': result, 'rule_trace': traces,
            'decisions': decisions, 'value_changing_corrections': []}

def entity_key(entity, concept=False):
    value = entity.get('rxcui') if concept and entity['type']=='drug_name' else None
    return (entity['medication_group'], entity['type'],
            str(value) if value else normalize_text(entity['value']))

def entity_counts(reference, prediction, concept=False):
    ref = Counter(entity_key(e, concept) for e in reference)
    pred = Counter(entity_key(e, concept) for e in prediction)
    matched = ref & pred
    per_type = {}
    for typ in ENTITY_TYPES:
        tp = sum(n for key,n in matched.items() if key[1] == typ)
        support = sum(n for key,n in ref.items() if key[1] == typ)
        asserted = sum(n for key,n in pred.items() if key[1] == typ)
        per_type[typ] = {'TP': tp, 'FP': asserted-tp,
                        'FN': support-tp, 'support': support,
                        'predictions': asserted}
    return {'per_type': per_type,
            'matched': [list(k)+[v] for k,v in matched.items()],
            'unmatched_reference': [list(k)+[v] for k,v in (ref-pred).items()],
            'unmatched_prediction': [list(k)+[v] for k,v in (pred-ref).items()],
            'record_exact': ref == pred}

def precision_recall_f1(tp, fp, fn):
    return {'precision': ratio(tp, tp+fp), 'recall': ratio(tp, tp+fn),
            'F1': ratio(2*tp, 2*tp+fp+fn)}

def summarize(rows, config_name):
    rows = [row for row in rows if row['config'] == config_name]
    chars, words = Counter(), Counter()
    types = {t: Counter() for t in ENTITY_TYPES}
    accepted = eligible_accepted = correct_accepted = ref_groups = exact = reviewed = failures = 0
    for row in rows:
        chars.update(row['text_scores']['character_counts'])
        words.update(row['text_scores']['word_counts'])
        for typ, counts in row['entity_scores']['per_type'].items():
            types[typ].update(counts)
        accepted += row['accepted_tuples']
        eligible_accepted += row['accepted_reference_groups']
        correct_accepted += row['correct_accepted_tuples']
        ref_groups += row['reference_tuple_count']
        exact += row['entity_scores']['record_exact']
        reviewed += row['document_review_required']
        failures += bool(row.get('ocr_error'))
    tp = sum(c['TP'] for c in types.values())
    fp = sum(c['FP'] for c in types.values())
    fn = sum(c['FN'] for c in types.values())
    return {'config': config_name, 'n_documents': len(rows),
        'micro_CER': ratio(chars['S']+chars['D']+chars['I'], chars['N']),
        'micro_WER': ratio(words['S']+words['D']+words['I'], words['N']),
        'macro_CER': safe_mean([r['text_scores']['CER'] for r in rows]),
        'macro_WER': safe_mean([r['text_scores']['WER'] for r in rows]),
        'TP': tp, 'FP': fp, 'FN': fn,
        **precision_recall_f1(tp, fp, fn),
        'record_exact_accuracy': ratio(exact, len(rows)),
        'field_exact_accuracy': ratio(tp, tp+fn),
        'accepted_tuples': accepted,
        'accepted_tuple_precision': ratio(correct_accepted, accepted),
        'accepted_incorrect_fraction': ratio(accepted-correct_accepted, accepted),
        'accepted_reference_groups': eligible_accepted,
        'coverage': ratio(eligible_accepted, ref_groups),
        'review_rate': ratio(reviewed, len(rows)),
        'ocr_failures': failures,
        'character_counts': dict(chars), 'word_counts': dict(words),
        'per_type': {t: {**dict(c), **precision_recall_f1(
            c['TP'], c['FP'], c['FN'])} for t,c in types.items()},
        'macro_F1_supported_types': safe_mean([
            precision_recall_f1(c['TP'], c['FP'], c['FN'])['F1']
            for c in types.values() if c['support']]),
    }

def make_r5_resource(entities, patient_ref, group_id, review_required):
    """Research drafts only; no invented patient identity or dose."""
    if not patient_ref or review_required:
        return None
    items = [e for e in entities if e['medication_group'] == group_id]
    drugs = [e for e in items if e['type'] == 'drug_name']
    if len(drugs) != 1:
        return None
    drug = drugs[0]
    strengths = [e['value'] for e in items if e['type'] == 'strength']
    concept = {'text': ' '.join([drug['value']] + strengths)}
    if drug.get('rxcui') and drug.get('canonical_display'):
        concept['coding'] = [{
            'system': 'http://www.nlm.nih.gov/research/umls/rxnorm',
            'code': drug['rxcui'], 'display': drug['canonical_display'],
        }]
    dosage_text = ' '.join(e['value'] for e in items
        if e['type'] in {'dose','frequency','route','duration','conditional'})
    resource = {'resourceType': 'MedicationRequest',
        'status': 'draft', 'intent': 'proposal',
        'subject': {'reference': patient_ref},
        'medication': {'concept': concept},
        'note': [{'text': 'Research draft; human review required.'},
                 {'text': 'Source fields: ' + '; '.join(
                     e['type'] + '=' + e['value'] for e in items)}]}
    supported = set(ENTITY_TYPES)
    unsupported = sorted({e['type'] for e in items} - supported)
    if unsupported:
        resource['note'].append({'text': 'No typed mapping; review: ' + ', '.join(unsupported)})
    if dosage_text:
        resource['dosageInstruction'] = [{'text': dosage_text}]
    return resource

def validate_r5_model(resource):
    """Package-model validation, distinct from HL7 profile validation."""
    from fhir.resources.medicationrequest import MedicationRequest
    try:
        model = MedicationRequest.model_validate(resource)
        return {'model_valid': True,
                'resource': model.model_dump(mode='json', exclude_none=True)}
    except Exception as error:
        return {'model_valid': False, 'error': str(error)}

def hl7_validate(resource_path, jar_path, jar_sha256, output_path):
    """Caller supplies a verified official validator JAR."""
    if digest_file(jar_path) != jar_sha256:
        raise ValueError('Validator JAR checksum mismatch')
    command = ['java','-jar',str(jar_path),str(resource_path),
               '-version','5.0.0','-output',str(output_path)]
    result = subprocess.run(command, capture_output=True, text=True,
                            timeout=300, check=False)
    return {'command': command, 'exit_code': result.returncode,
            'stdout': result.stdout, 'stderr': result.stderr,
            'report_path': str(output_path), 'jar_sha256': jar_sha256}

def crop_regions(row, base_dir):
    from PIL import Image
    image = Image.open(Path(base_dir)/row['image_path']).convert('RGB')
    regions = sorted(row['regions'], key=lambda r: r['order'])
    if not regions:
        raise ValueError('Explicit word/line regions required')
    for region in regions:
        x0,y0,x1,y1 = region['box']
        if not (0 <= x0 < x1 <= image.width and 0 <= y0 < y1 <= image.height):
            raise ValueError('Region outside original image')
        yield region, image.crop((x0,y0,x1,y1))

class OCRAdapters:
    """Same original crops for each recognizer; record model revision."""
    def __init__(self, config=CONFIG):
        self.config = config
        self.processor = self.model = self.easy_reader = None
        self.metadata = {}

    def tesseract(self, image):
        import pytesseract
        self.metadata['tesseract_version'] = str(pytesseract.get_tesseract_version())
        return pytesseract.image_to_string(image, config='--psm 7')

    def easyocr(self, image):
        import easyocr
        import numpy as np
        import torch
        if self.easy_reader is None:
            self.easy_reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
        return ' '.join(self.easy_reader.readtext(np.asarray(image), detail=0))

    def trocr(self, image):
        import torch
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        if self.model is None:
            torch.manual_seed(self.config.seed)
            self.processor = TrOCRProcessor.from_pretrained(
                self.config.checkpoint, revision=self.config.checkpoint_revision)
            self.model = VisionEncoderDecoderModel.from_pretrained(
                self.config.checkpoint, revision=self.config.checkpoint_revision)
            self.model.eval()
            self.model.to('cuda' if torch.cuda.is_available() else 'cpu')
            commit = getattr(self.model.config, '_commit_hash', None)
            self.metadata['resolved_checkpoint_revision'] = commit
        pixel_values = self.processor(images=image, return_tensors='pt').pixel_values
        device = next(self.model.parameters()).device
        with torch.inference_mode():
            tokens = self.model.generate(pixel_values.to(device),
                max_new_tokens=self.config.max_new_tokens,
                num_beams=self.config.num_beams)
        return self.processor.batch_decode(tokens, skip_special_tokens=True)[0]

def recognize_record(row, base_dir, ocr_function):
    region_outputs, errors = [], []
    started = time.perf_counter()
    for region, image in crop_regions(row, base_dir):
        try:
            text = ocr_function(image)
            error = None
        except Exception as exception:
            text, error = '', repr(exception)
            errors.append({'region_id': region['region_id'], 'error': error})
        region_outputs.append({**region, 'text': text, 'error': error})
    return {'regions': region_outputs, 'ocr_error': errors,
            'prediction': '\n'.join(r['text'] for r in region_outputs),
            'recognition_seconds': time.perf_counter()-started}

def score_configuration(row, recognition, aliases, client, use_rules, config=CONFIG):
    grouped_text = defaultdict(list)
    for region in recognition['regions']:
        grouped_text[region['medication_group']].append(region['text'])
    raw_entities = []
    for group, texts in grouped_text.items():
        raw_entities.extend(extract_group('\n'.join(texts), group, aliases))
    validation = validate_entities(raw_entities, client, config) if use_rules else {
        'entities': deepcopy(raw_entities), 'decisions': [],
        'rule_trace': [], 'value_changing_corrections': []}
    # Review missing/unrecognized groups instead of losing them.
    decisions = {x['medication_group']: x for x in validation['decisions']}
    for group in grouped_text:
        if group not in decisions:
            decisions[group] = {'medication_group': group,
                'review_required': use_rules, 'flags': ['no_entities'] if use_rules else []}
    candidates = validation['entities']
    pred = [e for e in candidates if not use_rules or
            not decisions[e['medication_group']]['review_required']]
    ref = row['reference_entities']
    ref_by_group = defaultdict(list)
    pred_by_group = defaultdict(list)
    for entity in ref:
        ref_by_group[entity['medication_group']].append(entity)
    for entity in pred:
        pred_by_group[entity['medication_group']].append(entity)
    accepted = eligible_accepted = correct = 0
    resources = []
    for group, entities in pred_by_group.items():
        review = decisions[group]['review_required']
        if not review:
            accepted += 1
            eligible_accepted += group in ref_by_group
            correct += entity_counts(ref_by_group[group], entities)['record_exact']
        resource = make_r5_resource(pred, row.get('research_patient_ref'), group, review)
        if resource is not None:
            resources.append(resource)
    concept_supported = all(e.get('rxcui') for e in ref if e['type']=='drug_name')
    return {'record_id': row['record_id'], 'source': row['source'],
        'language': row['language'], 'quality': row.get('quality','unclassified'),
        'group': row.get('writer_id') or row['prescription_group'],
        'prediction': recognition['prediction'], 'ocr_error': recognition['ocr_error'],
        'text_scores': text_scores(row['reference_text'], recognition['prediction']),
        'strict_text_scores': text_scores(row['reference_text'], recognition['prediction'], True),
        'raw_entities': raw_entities, 'candidate_entities': candidates,
        'pred_entities': pred,
        'entity_scores': entity_counts(ref, pred),
        'concept_entity_scores': entity_counts(ref, pred, concept=True) if concept_supported else None,
        'reference_tuple_count': len(ref_by_group),
        'accepted_tuples': accepted, 'correct_accepted_tuples': correct,
        'accepted_reference_groups': eligible_accepted,
        'document_review_required': any(x['review_required'] for x in decisions.values()),
        'decisions': list(decisions.values()), 'rule_trace': validation['rule_trace'],
        'value_changing_corrections': validation['value_changing_corrections'],
        'resources': resources, 'recognition_seconds': recognition['recognition_seconds'],
        'authentic': row['authentic'], 'split': row['split'], 'task': row['task']}

def run_evaluation(rows, base_dir, aliases, output_dir, config=CONFIG,
                   models=('tesseract','trocr')):
    validate_manifest(rows, base_dir)
    if 'trocr' in models and not re.fullmatch(r'[0-9a-f]{40}', config.checkpoint_revision):
        raise ValueError('Pin a 40-character model commit for final evaluation')
    test = [r for r in rows if r['authentic'] and r['split']=='test']
    if not test:
        raise ValueError('No adjudicated authentic held-out test observations')
    adapters = OCRAdapters(config)
    client = TerminologyClient(Path(output_dir)/'rxnorm_cache', config.allow_network)
    outputs = []
    for row in test:
        for model in models:
            recognition = recognize_record(row, base_dir, getattr(adapters, model))
            for use_rules in (False, True):
                result = score_configuration(row, recognition, aliases, client, use_rules, config)
                result['config'] = model + ('_rules' if use_rules else '_entities')
                result['fhir_model_reports'] = [
                    validate_r5_model(resource) for resource in result['resources']]
                result['hl7_profile_validation'] = 'not_run'
                outputs.append(result)
    output_dir = Path(output_dir)
    write_jsonl(output_dir/'predictions.jsonl', outputs)
    configs = sorted({r['config'] for r in outputs})
    summaries = [dict(task=task, **summarize(
        [r for r in outputs if r['task']==task], c))
        for task in sorted({r['task'] for r in outputs}) for c in configs]
    save_json(output_dir/'summary_results.json', summaries)
    flat = [{k:v for k,v in s.items() if not isinstance(v,dict)} for s in summaries]
    write_csv(output_dir/'vt_nspif_summary_results.csv', flat)
    save_json(output_dir/'run_manifest.json', {
        'started_or_saved_at': utc_now(), 'config': asdict(config),
        'code_sha256': digest_file(__file__),
        'manifest_sha256': digest_bytes(canonical_json(rows).encode()),
        'lexicon_sha256': digest_bytes(canonical_json(aliases).encode()),
        'python': platform.python_version(), 'platform': platform.platform(),
        'packages': {d.metadata['Name']:d.version for d in importlib.metadata.distributions()},
        'ocr_metadata': adapters.metadata,
        'terminology_version_response': client.get('version.json'),
        'evaluation_track': 'PRIMARY_AUTHENTIC_HELD_OUT',
        'segmentation_track': 'provided_regions',
        'training_condition': 'pretrained_inference_no_fine_tuning',
        'writer_grouping_complete': all(r.get('writer_id') for r in test),
    })
    return outputs, summaries

def percentile(values, fraction):
    values = sorted(values)
    if not values:
        return None
    position = (len(values)-1)*fraction
    lo, hi = math.floor(position), math.ceil(position)
    return values[lo] + (values[hi]-values[lo])*(position-lo)

def paired_bootstrap(rows, left, right, metric='F1', seed=42, n=2000):
    """Right minus left; resample writer/prescription groups jointly."""
    l_ids = {r['record_id'] for r in rows if r['config']==left}
    r_ids = {r['record_id'] for r in rows if r['config']==right}
    if l_ids != r_ids:
        raise ValueError('Paired configurations have different observations')
    groups = sorted({r['group'] for r in rows if r['config'] in {left,right}})
    if len(groups) < 2:
        raise ValueError('At least two independent groups needed for interval')
    grouped = {g:[r for r in rows if r['group']==g and r['config'] in {left,right}]
               for g in groups}
    rng, differences = random.Random(seed), []
    for _ in range(n):
        sample = [row for group in rng.choices(groups,k=len(groups))
                  for row in grouped[group]]
        a, b = summarize(sample,left)[metric], summarize(sample,right)[metric]
        if a is not None and b is not None:
            differences.append(b-a)
    a, b = summarize(rows,left)[metric], summarize(rows,right)[metric]
    return {'left': left, 'right': right, 'metric': metric,
        'effect_right_minus_left': b-a if a is not None and b is not None else None,
        'CI_95_low': percentile(differences,.025),
        'CI_95_high': percentile(differences,.975),
        'requested_replicates': n, 'defined_replicates': len(differences),
        'independent_groups': len(groups), 'seed': seed}

def subgroup_summaries(rows, by='source'):
    result = []
    for value in sorted({r[by] for r in rows}):
        subset = [r for r in rows if r[by]==value]
        for config in sorted({r['config'] for r in subset}):
            result.append({by:value, **summarize(subset,config)})
    return result

def export_chapter4(rows, output_dir):
    """Export only measured values, with denominators and track labels."""
    output_dir = Path(output_dir)
    tables = []
    for task in sorted({r['task'] for r in rows}):
        subset = [r for r in rows if r['task']==task]
        for config in sorted({r['config'] for r in subset}):
            s = summarize(subset,config)
            for typ, counts in s['per_type'].items():
                tables.append({'task':task, 'config':config,
                    'entity_type':typ, **counts,
                    'track':'PRIMARY_AUTHENTIC_HELD_OUT'})
    write_csv(output_dir/'chapter4_entity_metrics.csv',tables)
    save_json(output_dir/'chapter4_subgroups.json', {
        k:subgroup_summaries(rows,k) for k in ('source','language','quality')})
    effects = []
    for task in sorted({r['task'] for r in rows}):
        subset = [r for r in rows if r['task']==task]
        configs = {r['config'] for r in subset}
        if {'trocr_entities','trocr_rules'} <= configs:
            for metric in ('F1','micro_CER','micro_WER'):
                effects.append({'task':task, **paired_bootstrap(
                    subset,'trocr_entities','trocr_rules',metric)})
    save_json(output_dir/'chapter4_paired_effects.json',effects)
    return {'entity_rows':len(tables),'paired_effects':len(effects)}

def plot_measured_results(summary_path, output_path):
    import matplotlib.pyplot as plt
    rows = json.loads(Path(summary_path).read_text())
    eligible = [r for r in rows if r.get('micro_CER') is not None]
    if not eligible:
        raise ValueError('No measured CER values; do not draw placeholder chart')
    fig, axis = plt.subplots(figsize=(9,5))
    axis.bar([r.get('task','')+'\n'+r['config'] for r in eligible],
             [r['micro_CER'] for r in eligible])
    axis.set_ylabel('Corpus micro CER')
    axis.set_title('Authentic held-out recognition; provided regions')
    axis.tick_params(axis='x', rotation=30)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

def conformance_summary(rows):
    """Do not count package validity as completed HL7 conformance."""
    attempted = sum(len(r.get('resources',[])) for r in rows)
    model_valid = sum(report['model_valid'] for r in rows
                     for report in r.get('fhir_model_reports',[]))
    return {'attempted_resources':attempted,
            'package_model_valid_resources':model_valid,
            'package_model_valid_rate':ratio(model_valid,attempted),
            'hl7_profile_conformant_resources':None,
            'hl7_profile_validation_status':'not_run',
            'release':'5.0.0'}
