"""在线大模型分词：Prompt/Few-shot、HTTP 请求、严格校验、重试与缓存。"""
import json
import hashlib
import time
import urllib.error
import urllib.request
from urllib.parse import urlparse
from pathlib import Path
from config import load_settings
from storage import save


DEFAULT_BASE_URL = 'https://api.deepseek.com'
DEFAULT_MODEL = 'deepseek-flash'

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


def parse_response(content, text):
    content = content.strip()
    if content.startswith('```') and content.endswith('```'):
        content = content.split('\n', 1)[1].rsplit('```', 1)[0].strip()
    words = json.loads(content)
    if not isinstance(words, list) or any(not isinstance(w, str) or not w for w in words):
        raise ValueError('模型返回值必须为非空词字符串组成的 JSON 数组')
    returned = ''.join(words)
    if returned != text:
        position = next((i for i, (a, b) in enumerate(zip(text, returned)) if a != b), min(len(text), len(returned)))
        expected = text[position:position+1]
        actual = returned[position:position+1]
        context = text[max(0, position-12):position+13]
        raise ValueError(f'模型修改原文：字符位置 {position}（从0开始）应为 {expected!r}，却返回 {actual!r}；该处原文为 {context!r}；原文长 {len(text)}，返回长 {len(returned)}')
    return words


class LLMSegmenter:
    def __init__(self, training_examples=(), cache_dir=Path('Test1/results/llm_cache'),
                 timeout=90, retries=2, temperature=None, config_path=Path('.env')):
        settings = load_settings(config_path)
        self.base_url = settings.get('LLM_BASE_URL', DEFAULT_BASE_URL).rstrip('/')
        self.model = settings.get('LLM_MODEL', DEFAULT_MODEL)
        self.api_key = settings.get('LLM_API_KEY') or settings.get('DEEPSEEK_API_KEY', '')
        if not all((self.base_url, self.model, self.api_key)):
            raise ValueError('请在 NLP 根目录 .env 设置 LLM_API_KEY 或 DEEPSEEK_API_KEY；默认服务为 DeepSeek Flash')
        if not self.base_url.startswith('https://'):
            raise ValueError('在线 API Base URL 必须使用 HTTPS')
        self.examples = list(training_examples)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout, self.retries, self.temperature = timeout, retries, temperature
        self.last_info = {}

    def segment(self, text):
        messages = build_messages(text, self.examples)
        payload = {'model': self.model, 'messages': messages, 'stream': False}
        # 分词使用非思考模式，避免思考开销混入分词速度；通用服务不发送专有参数。
        if urlparse(self.base_url).hostname == 'api.deepseek.com':
            payload['thinking'] = {'type': 'disabled'}
        if self.temperature is not None:
            payload['temperature'] = self.temperature
        identity = json.dumps({'endpoint': self.base_url, 'payload': payload}, ensure_ascii=False, sort_keys=True)
        key = hashlib.sha256(identity.encode('utf-8')).hexdigest()
        path = self.cache_dir / (key + '.json')
        if path.exists():
            record = json.loads(path.read_text(encoding='utf-8'))
            words = parse_response(json.dumps(record['words'], ensure_ascii=False), text)
            self.last_info = {**record['info'], 'cache_hit': True}
            return words
        started = time.perf_counter()
        attempts = []
        diagnostic = self.cache_dir / (key + '.error.json')
        for attempt in range(self.retries+1):
            request = urllib.request.Request(self.base_url + '/chat/completions',
                data=json.dumps(payload, ensure_ascii=False).encode('utf-8'),
                headers={'Authorization': 'Bearer ' + self.api_key, 'Content-Type': 'application/json'}, method='POST')
            data, content = None, None
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    data = json.load(response)
                content = data['choices'][0]['message']['content']
                words = parse_response(content, text)
                info = {'cache_hit': False, 'request_seconds': time.perf_counter()-started,
                        'attempts': attempt+1, 'repair_attempts': attempts, 'usage': data.get('usage', {}), 'model': data.get('model', self.model)}
                path.write_text(json.dumps({'words': words, 'info': info}, ensure_ascii=False), encoding='utf-8')
                diagnostic.unlink(missing_ok=True)
                self.last_info = info
                return words
            except urllib.error.HTTPError as exc:
                reason = f'HTTP {exc.code}'
                attempts.append({'attempt': attempt+1, 'reason': reason})
                save(diagnostic, {'text_length': len(text), 'attempts': attempts})
                if exc.code not in (408, 429, 500, 502, 503, 504) or attempt == self.retries:
                    raise RuntimeError(f'在线 API {reason}，详情见 {diagnostic}') from None
            except (urllib.error.URLError, TimeoutError, ValueError, KeyError, IndexError, TypeError) as exc:
                # 不输出请求头、服务端错误正文或配置，避免泄露认证信息。
                if isinstance(exc, json.JSONDecodeError):
                    reason = '返回不是有效 JSON'
                elif isinstance(exc, ValueError):
                    reason = str(exc)
                else:
                    reason = type(exc).__name__
                attempts.append({'attempt': attempt+1, 'reason': reason,
                    'finish_reason': data.get('choices', [{}])[0].get('finish_reason') if isinstance(data, dict) and data.get('choices') else None,
                    'response': content if isinstance(content, str) else None})
                save(diagnostic, {'text_length': len(text), 'attempts': attempts})
                if attempt == self.retries:
                    raise RuntimeError(f'在线分词失败：{reason}；详情见 {diagnostic}') from None
                if isinstance(content, str):
                    # 只用原文校验信息要求模型纠错，不提供测试金标准。
                    payload['messages'] = messages + [
                        {'role': 'assistant', 'content': content},
                        {'role': 'user', 'content': '上次返回未通过校验：'+reason+'。请重新分词，只输出 JSON 字符串数组；必须逐字保留上面原文，尤其是全角数字与标点，不替换、不增删字符。'}]
            time.sleep(min(2**attempt, 8))
