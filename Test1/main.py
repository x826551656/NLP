"""实验一统一入口：预处理 → 训练词典 → 分词 → 评测与结果输出。"""
import argparse
import csv
import json
import platform
import random
import time
from pathlib import Path

from config import load_settings
from corpus import prepare_corpus
from dictionary import build_dictionary, fmm, rmm, bimm
from evaluate import evaluate_predictions, typical_errors, fertility
from hmm import HMMSegmenter
from bpe import BPETokenizer
from llm import LLMSegmenter
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
    training_seconds = {}
    print('Training HMM...', flush=True)
    started = time.perf_counter()
    hmm = HMMSegmenter().fit(row['words'] for row in train)
    training_seconds['HMM'] = time.perf_counter()-started
    hmm.save(out / 'hmm_model.json')
    print('Training BPE...', flush=True)
    started = time.perf_counter()
    bpe = BPETokenizer().fit((row['text'] for row in train), vocab_size=args.bpe_vocab_size, max_merges=args.bpe_merges)
    training_seconds['BPE'] = time.perf_counter()-started
    bpe.save(out / 'bpe_model.json')
    metrics, offline_predictions = [], {}
    methods = {'FMM': lambda text: fmm(text, vocab, maximum),
               'RMM': lambda text: rmm(text, vocab, maximum),
               'BiMM': lambda text: bimm(text, vocab, maximum),
               'HMM': hmm.segment, 'BPE': bpe.tokenize}
    for name, method in methods.items():
        start = time.perf_counter()
        predictions = [method(row['text']) for row in test]
        elapsed = time.perf_counter() - start
        metric, records, errors = evaluate_predictions(test, predictions, vocab, name, elapsed)
        offline_predictions[name] = predictions
        metric['training_seconds'] = training_seconds.get(name)
        metric['fertility'] = fertility(sum(map(len, predictions)), metric['gold_words']) if name == 'BPE' else None
        metrics.append(metric)
        jsonl(out / f'{name}_predictions.jsonl', records)
        jsonl(out / f'{name}_errors.jsonl', errors)
        jsonl(out / f'{name}_typical_errors.jsonl', typical_errors(errors))
    manifest = {
        'bpe_config': {'target_vocab_size': args.bpe_vocab_size, 'max_merges': args.bpe_merges, 'actual_vocab_size': len(bpe.vocabulary), 'actual_merges': len(bpe.merges)},
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
    write_summary(out, metrics, manifest)
    # 使用方向匹配分歧筛选案例，不预先查看测试标准答案选模型。
    disagreements = []
    for i, row in enumerate(test):
        if offline_predictions['FMM'][i] != offline_predictions['RMM'][i]:
            disagreements.append({'id': row['id'], 'text': row['text'], 'gold': row['words'],
                'predictions': {name: values[i] for name, values in offline_predictions.items()}})
    jsonl(out / 'direction_disagreement_cases.jsonl', disagreements)
    if args.with_llm:
        run_llm_comparison(args, train, test, vocab, methods, out)
    print(json.dumps({'manifest': manifest, 'metrics': metrics}, ensure_ascii=False, indent=2))


def write_summary(out, metrics, manifest):
    lines = ['# 离线实验真实结果', '',
        f"训练 {manifest['train_lines']} 行，测试 {manifest['test_lines']} 行；随机种子 {manifest['seed']}。", '',
        '|方法|P|R|F1|OOV召回|训练秒数|分词秒数|Fertility|',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for m in metrics:
        training = '-' if m['training_seconds'] is None else f"{m['training_seconds']:.4f}"
        fert = '-' if m['fertility'] is None else f"{m['fertility']:.4f}"
        lines.append(f"|{m['method']}|{m['precision']:.4%}|{m['recall']:.4%}|{m['f1']:.4%}|{m['oov_recall']:.4%}|{training}|{m['segmentation_seconds']:.4f}|{fert}|")
    lines.extend(['', '耗时为本机单次实测，训练与分词分别计时，不含评测和写文件。',
        'BPE 的词边界 P/R/F1 仅是与词级金标准的对齐诊断，不代表语言模型任务能力；主要粒度指标为 Fertility。',
        'HMM 与 BPE 均只使用训练集。在线大模型结果单独保存于 llm_comparison/，不能拿子集结果直接与本表全量结果比较。',
        '方向分歧案例见 direction_disagreement_cases.jsonl；各方法完整错误与典型错误分别保存。'])
    (out / 'summary.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def run_llm_comparison(args, train, test, vocab, methods, out):
    candidates = [row for row in train if 8 <= len(row['text']) <= 80]
    examples = candidates[:3]
    if len(examples) < 3:
        raise ValueError('缺少合适的训练集 Few-shot 示例')
    client = LLMSegmenter(examples, cache_dir=out / 'llm_cache', temperature=args.llm_temperature)
    limit = min(args.llm_limit, len(test))
    indices = sorted(random.Random(args.seed).sample(range(len(test)), limit))
    selected = [test[i] for i in indices]
    folder = out / 'llm_comparison'
    folder.mkdir(parents=True, exist_ok=True)
    jsonl(folder / 'test.jsonl', selected)
    jsonl(folder / 'few_shot_train.jsonl', examples)
    save(folder / 'run_config.json', {'model': client.model, 'base_url': client.base_url,
        'temperature': args.llm_temperature, 'thinking': 'disabled for api.deepseek.com', 'seed': args.seed, 'test_lines': limit,
        'status': 'in_progress', 'timing_policy': 'network_seconds excludes cached requests; cached original latencies are historical'})
    # 不让旧的完成指标冒充本次尚未完成的在线实验。
    (folder / 'metrics.json').unlink(missing_ok=True)
    predictions, request_info = [], []
    started = time.perf_counter()
    for i, row in enumerate(selected, 1):
        try:
            predictions.append(client.segment(row['text']))
        except RuntimeError as exc:
            jsonl(folder / 'request_info.jsonl', request_info)
            jsonl(folder / 'LLM_partial_predictions.jsonl', [{'id': r['id'], 'predicted': pred} for r, pred in zip(selected, predictions)])
            save(folder / 'failure.json', {'position': i, 'id': row['id'], 'text_length': len(row['text']), 'completed': len(predictions), 'reason': str(exc)})
            save(folder / 'run_config.json', {'model': client.model, 'seed': args.seed, 'test_lines': limit, 'completed': len(predictions), 'status': 'failed'})
            raise RuntimeError(f'第 {i}/{limit} 条失败，已完成结果和缓存已保留。{exc}') from None
        request_info.append({'id': row['id'], **client.last_info})
        print(f'LLM {i}/{limit}; cache_hit={client.last_info["cache_hit"]}', flush=True)
    elapsed = time.perf_counter()-started
    jsonl(folder / 'request_info.jsonl', request_info)
    all_metrics = []
    for name, method in methods.items():
        start = time.perf_counter()
        subset_predictions = [method(row['text']) for row in selected]
        subset_seconds = time.perf_counter()-start
        metric, records, errors = evaluate_predictions(selected, subset_predictions, vocab, name, subset_seconds)
        metric['timing_note'] = 'fresh segmentation on the identical LLM subset'
        metric['fertility'] = fertility(metric['predicted_words'], metric['gold_words']) if name == 'BPE' else None
        all_metrics.append(metric)
        jsonl(folder / f'{name}_predictions.jsonl', records)
    metric, records, errors = evaluate_predictions(selected, predictions, vocab, 'LLM', elapsed)
    metric['cache_hits'] = sum(info['cache_hit'] for info in request_info)
    metric['network_seconds'] = sum(info['request_seconds'] for info in request_info if not info['cache_hit'])
    metric['timing_note'] = 'segmentation_seconds includes cache I/O and retries; network_seconds excludes cache hits'
    all_metrics.append(metric)
    (folder / 'failure.json').unlink(missing_ok=True)
    (folder / 'LLM_partial_predictions.jsonl').unlink(missing_ok=True)
    jsonl(folder / 'LLM_predictions.jsonl', records)
    jsonl(folder / 'LLM_errors.jsonl', errors)
    jsonl(folder / 'LLM_typical_errors.jsonl', typical_errors(errors))
    save(folder / 'metrics.json', all_metrics)
    save(folder / 'run_config.json', {'model': client.model, 'base_url': client.base_url,
        'temperature': args.llm_temperature, 'thinking': 'disabled for api.deepseek.com', 'seed': args.seed, 'test_lines': limit, 'status': 'complete'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--segmented', type=Path, default=EXPERIMENT_DIR / 'data/1998人民日报（分词）.txt')
    parser.add_argument('--pos', type=Path, default=EXPERIMENT_DIR / 'data/1998人民日报（带词性标注）.txt')
    parser.add_argument('--output', type=Path, default=EXPERIMENT_DIR / 'results')
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--bpe-vocab-size', type=int, default=8000)
    parser.add_argument('--bpe-merges', type=int, default=500)
    parser.add_argument('--with-llm', action='store_true', help='显式启用在线 API')
    parser.add_argument('--llm-limit', type=int, default=100, help='所有方法统一比较的在线测试子集大小')
    parser.add_argument('--llm-temperature', type=float, default=None)
    args = parser.parse_args()
    if args.llm_limit <= 0:
        parser.error('--llm-limit 必须为正')
    if args.with_llm:
        settings = load_settings()
        if not (settings.get('LLM_API_KEY') or settings.get('DEEPSEEK_API_KEY')):
            parser.error('请在 NLP 根目录 .env 设置 LLM_API_KEY 或 DEEPSEEK_API_KEY；默认模型 deepseek-flash')
    run(args)
