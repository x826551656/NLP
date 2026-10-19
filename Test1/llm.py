"""在线大模型分词的 Prompt 与 Few-shot 构建；本阶段不调用 API。"""
import json


SYSTEM_PROMPT = (
    '你是中文分词工具。按词级粒度切分用户文本，保留所有字符、标点、数字。'
    '只返回 JSON 字符串数组，不添加解释；数组元素拼接必须等于原文。'
)


def build_messages(text, training_examples=()):
    """training_examples 必须由调用者仅从训练集选取。"""
    messages = [{'role': 'system', 'content': SYSTEM_PROMPT}]
    for example in training_examples:
        if ''.join(example['words']) != example['text']:
            raise ValueError('Few-shot 标注与文本不一致')
        messages.extend([
            {'role': 'user', 'content': example['text']},
            {'role': 'assistant', 'content': json.dumps(example['words'], ensure_ascii=False)},
        ])
    messages.append({'role': 'user', 'content': text})
    return messages


class LLMSegmenter:
    def segment(self, text):
        raise NotImplementedError('在线 API 尚未接入，本阶段不发送网络请求')
