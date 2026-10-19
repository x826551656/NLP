"""Reproducible dictionary segmentation experiment; Python 3.10+, stdlib only."""
import argparse
import csv
import hashlib
import json
import math
import platform
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path


def save(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def jsonl(path, rows):
    with path.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


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


def segment(text, vocab, maximum, reverse=False):
    result = []
    if not reverse:
        start = 0
        while start < len(text):
            end = min(len(text), start + maximum)
            while end > start + 1 and text[start:end] not in vocab:
                end -= 1
            result.append(text[start:end])
            start = end
    else:
        end = len(text)
        while end:
            start = max(0, end - maximum)
            while start < end - 1 and text[start:end] not in vocab:
                start += 1
            result.append(text[start:end])
            end = start
        result.reverse()
    return result


def bimm(text, vocab, maximum):
    forward = segment(text, vocab, maximum)
    backward = segment(text, vocab, maximum, True)
    # Fewer words, then fewer single-character words; final tie chooses FMM.
    key = lambda words: (len(words), sum(len(w) == 1 for w in words))
    return backward if key(backward) < key(forward) else forward


def spans(words):
    start = 0
    result = set()
    for word in words:
        result.add((start, start + len(word)))
        start += len(word)
    return result


def self_test():
    vocab = {'研究', '研究生', '生命', '命', '起源'}
    assert segment('研究生命起源', vocab, 3) == ['研究生', '命', '起源']
    assert segment('研究生命起源', vocab, 3, True) == ['研究', '生命', '起源']
    assert bimm('研究生命起源', vocab, 3) == ['研究', '生命', '起源']
    assert segment('陌生', set(), 1) == ['陌', '生']
    assert spans(['中国', '人']) & spans(['中', '国人']) == set()
    assert pos_words('[中央/n 人民/n 广播/vn 电台/n]nt') == ['中央', '人民', '广播', '电台']


def run(args):
    self_test()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    lines, source = read_source(args.segmented, 'gbk')
    tagged, pos_source = read_source(args.pos, 'utf-8')
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
    random.Random(args.seed).shuffle(rows)
    cut = math.floor(len(rows) * 0.8)
    train, test = rows[:cut], rows[cut:]
    assert train and test
    assert not ({r['text'] for r in train} & {r['text'] for r in test})
    frequency = Counter(w for row in train for w in row['words'])
    vocab = set(frequency)
    assert vocab == {w for row in train for w in row['words']}
    maximum = max(map(len, vocab))
    jsonl(out / 'train.jsonl', train)
    jsonl(out / 'test.jsonl', test)
    jsonl(out / 'annotation_conflicts.jsonl', conflicts)
    jsonl(out / 'pos_audit.jsonl', audit)
    with (out / 'dictionary.tsv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(['word', 'train_frequency'])
        writer.writerows(sorted(frequency.items()))
    total_gold = sum(len(r['words']) for r in test)
    oov_total = sum(w not in vocab for r in test for w in r['words'])
    metrics = []
    for name, fn in [('FMM', lambda t: segment(t, vocab, maximum)),
                     ('RMM', lambda t: segment(t, vocab, maximum, True)),
                     ('BiMM', lambda t: bimm(t, vocab, maximum))]:
        start = time.perf_counter()
        predictions = [fn(row['text']) for row in test]
        elapsed = time.perf_counter() - start  # Excludes evaluation and file I/O.
        correct = predicted = exact = oov_correct = 0
        errors, records = [], []
        for row, pred in zip(test, predictions):
            assert ''.join(pred) == row['text']
            gold_spans, pred_spans = spans(row['words']), spans(pred)
            correct += len(gold_spans & pred_spans)
            predicted += len(pred)
            exact += pred == row['words']
            offset = 0
            oov = []
            for word in row['words']:
                if word not in vocab:
                    oov.append(word)
                    oov_correct += (offset, offset + len(word)) in pred_spans
                offset += len(word)
            record = {'id': row['id'], 'source_lines': row['source_lines'],
                      'text': row['text'], 'gold': row['words'], 'predicted': pred,
                      'oov_words': oov, 'boundary_errors': len(gold_spans ^ pred_spans)}
            records.append(record)
            if pred != row['words']:
                errors.append(record)
        p, r = correct / predicted, correct / total_gold
        metric = {'method': name, 'precision': p, 'recall': r, 'f1': 2*p*r/(p+r),
                  'correct_words': correct, 'predicted_words': predicted,
                  'gold_words': total_gold, 'exact_lines': exact,
                  'test_lines': len(test), 'oov_words': oov_total,
                  'oov_correct': oov_correct, 'oov_recall': oov_correct/oov_total if oov_total else None,
                  'segmentation_seconds': elapsed,
                  'characters_per_second': sum(len(r['text']) for r in test)/elapsed}
        metrics.append(metric)
        jsonl(out / f'{name}_predictions.jsonl', records)
        jsonl(out / f'{name}_errors.jsonl', errors)
        # Deterministic, readable examples: shortest errors with/without OOV.
        selected = []
        for has_oov in [False, True]:
            subset = sorted((e for e in errors if bool(e['oov_words']) == has_oov),
                            key=lambda e: (len(e['text']), e['id']))
            selected.extend(subset[:10])
        jsonl(out / f'{name}_typical_errors.jsonl', selected)
    manifest = {'seed': args.seed, 'python': platform.python_version(),
                'sources': [source, pos_source], 'blank_lines': sum(not l.strip() for l in lines),
                'unique_texts_before_conflict_removal': len(grouped),
                'duplicate_extra_lines': sum(len(e)-1 for e in grouped.values()),
                'conflicting_texts_excluded': len(conflicts), 'pos_mismatch_lines': len(audit),
                'retained_lines': len(rows), 'train_lines': len(train), 'test_lines': len(test),
                'dictionary_size': len(vocab), 'max_word_length': maximum,
                'train_test_text_overlap': 0, 'test_oov_rate': oov_total/total_gold,
                'split_policy': 'deduplicate raw text; exclude conflicting annotations; seeded shuffle; floor(0.8*N) train',
                'metric_policy': 'micro exact word spans [start,end), punctuation included; line-local offsets',
                'pos_policy': 'audit only, never add POS words or external dictionary to training'}
    save(out / 'manifest.json', manifest)
    save(out / 'metrics.json', metrics)
    with (out / 'metrics.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics[0]))
        writer.writeheader(); writer.writerows(metrics)
    print(json.dumps({'manifest': manifest, 'metrics': metrics}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--segmented', type=Path, required=True)
    parser.add_argument('--pos', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path(__file__).parent / 'results')
    parser.add_argument('--seed', type=int, default=2026)
    run(parser.parse_args())
