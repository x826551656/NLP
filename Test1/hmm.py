"""后续阶段：BMES 状态 HMM 参数训练与 Viterbi 解码接口。"""


class HMMSegmenter:
    """尚未实现；训练时只能使用 data/train.jsonl。"""

    def fit(self, train_words):
        raise NotImplementedError('HMM 参数训练将在后续阶段实现')

    def segment(self, text):
        raise NotImplementedError('Viterbi 解码将在后续阶段实现')
