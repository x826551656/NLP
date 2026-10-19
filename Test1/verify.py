"""Independently verify saved dictionary, splits and micro span counts."""
import csv
import json
from collections import Counter
from pathlib import Path

base = Path(__file__).parent / 'results'
load = lambda name: [json.loads(s) for s in (base / name).read_text(encoding='utf-8').splitlines()]
train, test = load('train.jsonl'), load('test.jsonl')
assert len(train) == int((len(train) + len(test)) * 0.8)
assert not ({r['text'] for r in train} & {r['text'] for r in test})
assert len({r['text'] for r in train + test}) == len(train + test)
expected = Counter(w for r in train for w in r['words'])
with (base / 'dictionary.tsv').open(encoding='utf-8', newline='') as f:
    actual = {r['word']: int(r['train_frequency']) for r in csv.DictReader(f, delimiter='\t')}
assert actual == dict(expected)
for metric in json.loads((base / 'metrics.json').read_text(encoding='utf-8')):
    predictions = load(metric['method'] + '_predictions.jsonl')
    assert len(predictions) == len(test)
    hits = gold_count = pred_count = 0
    for gold, prediction in zip(test, predictions):
        assert gold['id'] == prediction['id']
        assert gold['words'] == prediction['gold']
        assert ''.join(prediction['predicted']) == gold['text']
        def boundaries(words):
            end = 0
            for word in words:
                begin = end
                end += len(word)
                yield begin, end
        reference = set(boundaries(gold['words']))
        output = set(boundaries(prediction['predicted']))
        hits += len(reference.intersection(output))
        gold_count += len(reference)
        pred_count += len(output)
    assert (hits, gold_count, pred_count) == (metric['correct_words'], metric['gold_words'], metric['predicted_words'])
    assert abs(metric['f1'] - 2*hits/(gold_count+pred_count)) < 1e-12
print('PASS: unique disjoint 80/20 split; train-only word frequencies; all saved predictions and span metrics.')
