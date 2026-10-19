"""字符 BPE：仅对训练原文统计频次，用堆和倒排索引更新合并对。"""
import heapq
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


def chunks(text):
    # 相同的无监督预切分用于训练和测试；不读取测试标准词边界。
    return re.findall(r'[\u3400-\u9fff]+|[A-Za-z0-9０-９]+|[\s\S]', text)


def merge(sequence, pair):
    result, i = [], 0
    while i < len(sequence):
        if i+1 < len(sequence) and (sequence[i], sequence[i+1]) == pair:
            result.append(sequence[i] + sequence[i+1])
            i += 2
        else:
            result.append(sequence[i])
            i += 1
    return tuple(result)


class BPETokenizer:
    def fit(self, train_texts, vocab_size=8000, max_merges=500, min_frequency=2):
        if vocab_size < 1 or max_merges < 0 or min_frequency < 1:
            raise ValueError('BPE 参数不合法')
        frequency = Counter(tuple(chunk) for text in train_texts for chunk in chunks(text))
        if not frequency:
            raise ValueError('BPE 训练文本不能为空')
        self.vocabulary = {c for sequence in frequency for c in sequence}
        sequences, weights = list(frequency), list(frequency.values())
        counts, index = Counter(), defaultdict(set)
        for i, sequence in enumerate(sequences):
            for pair, n in Counter(zip(sequence, sequence[1:])).items():
                counts[pair] += n*weights[i]
                index[pair].add(i)
        heap = [(-n, pair) for pair, n in counts.items()]
        heapq.heapify(heap)
        self.merges = []
        for _ in range(max_merges):
            if len(self.vocabulary) >= vocab_size:
                break
            while heap:
                negative, pair = heapq.heappop(heap)
                if -negative == counts[pair]:
                    break
            else:
                break
            if counts[pair] < min_frequency:
                break
            affected = list(index[pair])
            changed = set()
            for i in affected:
                previous = sequences[i]
                updated = merge(previous, pair)
                for p, n in Counter(zip(previous, previous[1:])).items():
                    counts[p] -= n*weights[i]
                    index[p].discard(i)
                    changed.add(p)
                for p, n in Counter(zip(updated, updated[1:])).items():
                    counts[p] += n*weights[i]
                    index[p].add(i)
                    changed.add(p)
                sequences[i] = updated
            for p in changed:
                if counts[p] > 0:
                    heapq.heappush(heap, (-counts[p], p))
            self.merges.append(pair)
            self.vocabulary.add(''.join(pair))
        self.ranks = {pair: i for i, pair in enumerate(self.merges)}
        return self

    def tokenize(self, text):
        if not hasattr(self, 'ranks'):
            raise RuntimeError('请先训练或加载 BPE')
        result = []
        for chunk in chunks(text):
            sequence = tuple(chunk)
            while len(sequence) > 1:
                candidates = [(self.ranks[p], p) for p in zip(sequence, sequence[1:]) if p in self.ranks]
                if not candidates:
                    break
                sequence = merge(sequence, min(candidates)[1])
            result.extend(sequence)
        # 未见字符保留为原字符，防止 UNK 丢失原文。
        return result

    def save(self, path):
        Path(path).write_text(json.dumps({'merges': self.merges, 'vocabulary': sorted(self.vocabulary)}, ensure_ascii=False), encoding='utf-8')

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text(encoding='utf-8'))
        obj = cls()
        obj.merges = [tuple(p) for p in data['merges']]
        obj.vocabulary = set(data['vocabulary'])
        obj.ranks = {pair: i for i, pair in enumerate(obj.merges)}
        return obj
