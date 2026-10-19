"""训练词典及正向、逆向、双向最大匹配。仅传入训练集构词典。"""
from collections import Counter


def build_dictionary(train):
    frequency = Counter(word for row in train for word in row['words'])
    if not frequency:
        raise ValueError('训练集词典不能为空')
    return frequency


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


def fmm(text, vocab, maximum):
    return segment(text, vocab, maximum)


def rmm(text, vocab, maximum):
    return segment(text, vocab, maximum, reverse=True)
