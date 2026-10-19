"""监督训练的 BMES HMM；仅允许合法状态转移，使用对数 Viterbi 解码。"""
import json
import math
from collections import Counter
from pathlib import Path

STATES = ('B', 'M', 'E', 'S')
NEXT = {'B': ('M', 'E'), 'M': ('M', 'E'), 'E': ('B', 'S'), 'S': ('B', 'S')}


class HMMSegmenter:
    def fit(self, train_words, alpha=0.1):
        if alpha <= 0:
            raise ValueError('alpha 必须为正')
        starts = Counter()
        transitions = {s: Counter() for s in STATES}
        emissions = {s: Counter() for s in STATES}
        vocabulary = set()
        count = 0
        for words in train_words:
            chars, tags = [], []
            for word in words:
                if not word:
                    raise ValueError('训练词不能为空')
                chars.extend(word)
                tags.extend(['S'] if len(word) == 1 else ['B'] + ['M']*(len(word)-2) + ['E'])
            if not chars:
                continue
            count += 1
            starts[tags[0]] += 1
            for ch, tag in zip(chars, tags):
                emissions[tag][ch] += 1
                vocabulary.add(ch)
            for a, b in zip(tags, tags[1:]):
                transitions[a][b] += 1
        if not count:
            raise ValueError('训练数据不能为空')
        self.start = {s: math.log((starts[s]+alpha)/(count+2*alpha)) for s in ('B', 'S')}
        self.transition = {}
        self.emission, self.unknown = {}, {}
        for s in STATES:
            total = sum(transitions[s].values()) + alpha*len(NEXT[s])
            self.transition[s] = {t: math.log((transitions[s][t]+alpha)/total) for t in NEXT[s]}
            total = sum(emissions[s].values()) + alpha*(len(vocabulary)+1)
            self.unknown[s] = math.log(alpha/total)
            self.emission[s] = {c: math.log((n+alpha)/total) for c, n in emissions[s].items()}
        return self

    def segment(self, text):
        if not hasattr(self, 'start'):
            raise RuntimeError('请先训练或加载 HMM')
        if not text:
            return []
        score = {s: self.start.get(s, -math.inf) + self.emission[s].get(text[0], self.unknown[s]) for s in STATES}
        backtrace = []
        for ch in text[1:]:
            current, previous = {}, {}
            for s in STATES:
                candidates = [(score[p] + self.transition[p][s], p) for p in STATES if s in NEXT[p]]
                value, parent = max(candidates)
                current[s] = value + self.emission[s].get(ch, self.unknown[s])
                previous[s] = parent
            backtrace.append(previous)
            score = current
        last = max(('E', 'S'), key=lambda s: score[s])
        tags = [last]
        for previous in reversed(backtrace):
            tags.append(previous[tags[-1]])
        tags.reverse()
        result, start = [], 0
        for end, tag in enumerate(tags, 1):
            if tag in ('E', 'S'):
                result.append(text[start:end])
                start = end
        assert start == len(text)
        return result

    def save(self, path):
        Path(path).write_text(json.dumps({k: getattr(self, k) for k in ('start', 'transition', 'emission', 'unknown')}, ensure_ascii=False), encoding='utf-8')

    @classmethod
    def load(cls, path):
        obj = cls()
        for k, v in json.loads(Path(path).read_text(encoding='utf-8')).items():
            setattr(obj, k, v)
        return obj
