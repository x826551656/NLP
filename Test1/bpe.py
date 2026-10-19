"""后续阶段：训练集学习 BPE 合并规则并切分测试原文。"""


class BPETokenizer:
    """尚未实现；训练时只能使用训练文本。"""

    def fit(self, train_texts, vocab_size=8000):
        raise NotImplementedError('BPE 词表训练将在后续阶段实现')

    def tokenize(self, text):
        raise NotImplementedError('BPE 子词切分将在后续阶段实现')
