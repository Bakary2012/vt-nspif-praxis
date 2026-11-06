"""Source-label word benchmark; not clinical entity adjudication."""
from pathlib import Path
from collections import Counter
from dataclasses import replace
import csv, hashlib, json, time, zipfile, re
import vt_nspif_core as vt

def unpack(archive, destination):
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        for member in z.infolist():
            target = (destination / member.filename).resolve()
            if not target.is_relative_to(destination):
                raise ValueError('Unsafe archive path')
            if (member.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError('Archive symlink rejected')
        z.extractall(destination)

def load_split(root, split, image_column=None, text_column=None):
    root = Path(root)
    labels = [p for p in root.rglob('*.csv')
              if re.sub(r'[^a-z]', '', p.stem.lower()) in {split+'label',split+'labels'}]
    if len(labels) != 1:
        raise ValueError(f'Expected one {split}_labels.csv; found {len(labels)}')
    with labels[0].open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames or []
        def column(explicit, candidates):
            hits = [f for f in fields if f.lower().strip() in candidates]
            if explicit in fields: return explicit
            if explicit or len(hits) != 1:
                raise ValueError(f'Set column override; available columns: {fields}')
            return hits[0]
        ic = column(image_column, {'images','image','image_name','image_path','filename','file_name','file','img','image_file'})
        tc = column(text_column, {'text','label','word','transcription','ground_truth','medicine_name'})
        records = list(reader)
    images = [p for p in root.rglob('*') if p.suffix.lower() in {'.png','.jpg','.jpeg','.bmp','.tif','.tiff'}]
    by_name = {}
    for p in images: by_name.setdefault(p.name, []).append(p)
    output, seen = [], set()
    for row in records:
        name, reference = row[ic].strip().replace('\\','/'), row[tc].strip()
        candidates = [p for p in images if p.as_posix().endswith('/'+name)]
        if not candidates: candidates = by_name.get(Path(name).name, [])
        if len(candidates) != 1 or not reference:
            raise ValueError(f'Ambiguous/missing image or empty reference: {name}')
        path = candidates[0]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest in seen: raise ValueError(f'Duplicate image content in {split}: {name}')
        seen.add(digest)
        output.append({'record_id':path.name,'image_path':str(path.resolve()),
            'reference_text':reference,'image_sha256':digest,'split':split,
            'scorable_reference':not bool(re.fullmatch(r'[?\s]+',reference))})
    return output, hashlib.sha256(labels[0].read_bytes()).hexdigest()
