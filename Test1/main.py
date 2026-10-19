"""实验一统一入口：预处理 → 训练词典 → 分词 → 评测与结果输出。"""
import argparse
import csv
import json
import platform
import time
from pathlib import Path

from corpus import prepare_corpus
from dictionary import build_dictionary, fmm, rmm, bimm
from evaluate import evaluate_predictions, typical_errors
from storage import save, jsonl

# 从 NLP 项目根目录运行，所有默认数据路径相对于该工作目录。
EXPERIMENT_DIR = Path('Test1')


def run(args):
    if not (EXPERIMENT_DIR / 'main.py').is_file():
        raise SystemExit('请从 NLP 项目根目录运行：python Test1/main.py')
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    train, test, conflicts, audit, stats = prepare_corpus(args.segmented, args.pos, args.seed)
    frequency = build_dictionary(train)
    vocab = set(frequency)
    maximum = max(map(len, vocab))
    # The canonical split lives under data/; results/ copies preserve prior outputs.
    for destination in {EXPERIMENT_DIR / 'data', out}:
        jsonl(destination / 'train.jsonl', train)
        jsonl(destination / 'test.jsonl', test)
    jsonl(out / 'annotation_conflicts.jsonl', conflicts)
    jsonl(out / 'pos_audit.jsonl', audit)
    with (out / 'dictionary.tsv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(['word', 'train_frequency'])
        writer.writerows(sorted(frequency.items()))
    metrics = []
    for name, method in [('FMM', fmm), ('RMM', rmm), ('BiMM', bimm)]:
        start = time.perf_counter()
        predictions = [method(row['text'], vocab, maximum) for row in test]
        elapsed = time.perf_counter() - start
        metric, records, errors = evaluate_predictions(test, predictions, vocab, name, elapsed)
        metrics.append(metric)
        jsonl(out / f'{name}_predictions.jsonl', records)
        jsonl(out / f'{name}_errors.jsonl', errors)
        jsonl(out / f'{name}_typical_errors.jsonl', typical_errors(errors))
    manifest = {
        'seed': args.seed, 'python': platform.python_version(), **stats,
        'train_lines': len(train), 'test_lines': len(test),
        'dictionary_size': len(vocab), 'max_word_length': maximum,
        'train_test_text_overlap': 0,
        'test_oov_rate': metrics[0]['oov_words']/metrics[0]['gold_words'],
        'split_policy': 'deduplicate raw text; exclude conflicting annotations; seeded shuffle; floor(0.8*N) train',
        'metric_policy': 'micro exact word spans [start,end), punctuation included; line-local offsets',
        'pos_policy': 'audit only, never add POS words or external dictionary to training',
    }
    save(out / 'manifest.json', manifest)
    save(out / 'metrics.json', metrics)
    with (out / 'metrics.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)
    print(json.dumps({'manifest': manifest, 'metrics': metrics}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--segmented', type=Path, default=EXPERIMENT_DIR / 'data/1998人民日报（分词）.txt')
    parser.add_argument('--pos', type=Path, default=EXPERIMENT_DIR / 'data/1998人民日报（带词性标注）.txt')
    parser.add_argument('--output', type=Path, default=EXPERIMENT_DIR / 'results')
    parser.add_argument('--seed', type=int, default=2026)
    run(parser.parse_args())
