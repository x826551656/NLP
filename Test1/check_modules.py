"""模块边界及分词评测的人工样例检查。"""
from corpus import pos_words
from dictionary import segment, bimm
from evaluate import spans, fertility
from llm import build_messages


def self_test():
    vocab = {'研究', '研究生', '生命', '命', '起源'}
    assert segment('研究生命起源', vocab, 3) == ['研究生', '命', '起源']
    assert segment('研究生命起源', vocab, 3, True) == ['研究', '生命', '起源']
    assert bimm('研究生命起源', vocab, 3) == ['研究', '生命', '起源']
    assert segment('陌生', set(), 1) == ['陌', '生']
    assert spans(['中国', '人']) & spans(['中', '国人']) == set()
    assert pos_words('[中央/n 人民/n 广播/vn 电台/n]nt') == ['中央', '人民', '广播', '电台']


if __name__ == '__main__':
    self_test()
    assert fertility(6, 3) == 2
    messages = build_messages('研究生命', [{'text': '中文', 'words': ['中文']}])
    assert len(messages) == 4 and messages[-1]['content'] == '研究生命'
    print('PASS: matching, BiMM, spans, POS parsing, Fertility and few-shot messages.')
