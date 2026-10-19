"""严格解码、清洗去重、语料审计和固定随机划分。"""
import hashlib
import math
import random
import re
from collections import defaultdict


def read_source(path, encoding):
    raw = path.read_bytes()
    text = raw.decode(encoding, errors='strict')
    assert text.encode(encoding) == raw
    if '\ufffd' in text:
        raise ValueError('Replacement character in source')
    return text.splitlines(), {
        'file': path.name, 'encoding': encoding, 'bytes': len(raw),
        'sha256': hashlib.sha256(raw).hexdigest(), 'lines': len(text.splitlines()),
        'prefix': text[:400],
    }

def pos_words(line):
    result = []
    for token in line.split():
        # Keep component words inside [word/tag ... word/tag]group_tag.
        word, tag = token.rsplit('/', 1)
        if word.startswith('['):
            word = word[1:]
        if not word or not re.fullmatch(r'[A-Za-z]+(?:\][A-Za-z]+)?', tag):
            raise ValueError(f'Unexpected POS token: {token}')
        result.append(word)
    return result


def prepare_corpus(segmented, pos, seed=2026):
    lines, source = read_source(segmented, 'gbk')
    tagged, pos_source = read_source(pos, 'utf-8')
    grouped = defaultdict(list)
    for number, line in enumerate(lines, 1):
        words = line.split()  # Normalize separators only; preserve every character.
        if words:
            grouped[''.join(words)].append({'line': number, 'words': words})
    conflicts, rows = [], []
    for text, entries in grouped.items():
        if len({tuple(e['words']) for e in entries}) > 1:
            conflicts.append({'text': text, 'annotations': entries})
        else:
            rows.append({'id': hashlib.sha256(text.encode('utf-8')).hexdigest(),
                         'text': text, 'words': entries[0]['words'],
                         'source_lines': [e['line'] for e in entries]})
    audit = []
    for i in range(max(len(lines), len(tagged))):
        try:
            a = lines[i].split() if i < len(lines) else []
            b = pos_words(tagged[i]) if i < len(tagged) else []
            if a != b:
                audit.append({'line': i + 1, 'same_text': ''.join(a) == ''.join(b),
                              'segmented': a, 'pos_stripped': b})
        except ValueError as exc:
            audit.append({'line': i + 1, 'parse_error': str(exc)})
    random.Random(seed).shuffle(rows)
    cut = math.floor(len(rows) * 0.8)
    train, test = rows[:cut], rows[cut:]
    assert train and test
    assert not ({r['text'] for r in train} & {r['text'] for r in test})
    stats = {
        'sources': [source, pos_source],
        'blank_lines': sum(not line.strip() for line in lines),
        'unique_texts_before_conflict_removal': len(grouped),
        'duplicate_extra_lines': sum(len(entries)-1 for entries in grouped.values()),
        'conflicting_texts_excluded': len(conflicts),
        'pos_mismatch_lines': len(audit),
        'retained_lines': len(rows),
    }
    return train, test, conflicts, audit, stats
