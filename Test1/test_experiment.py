"""关键算法的独立小规模核验；API 测试使用模拟响应，不访问网络。"""
import itertools
import json
import math
import os
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

from bpe import BPETokenizer, chunks, merge
from hmm import HMMSegmenter, STATES, NEXT
from llm import LLMSegmenter, parse_response
from evaluate import evaluate_predictions, fertility


class ExperimentTests(unittest.TestCase):
    def test_viterbi_against_exhaustive_search(self):
        model = HMMSegmenter().fit([['甲乙', '丙'], ['甲', '乙丙'], ['甲乙丙'], ['甲', '乙', '丙']])
        for text in ['甲', '甲乙', '甲乙丙', '甲乙陌丙']:
            candidates = []
            for tags in itertools.product(STATES, repeat=len(text)):
                if tags[0] not in ('B', 'S') or tags[-1] not in ('E', 'S'):
                    continue
                if any(b not in NEXT[a] for a, b in zip(tags, tags[1:])):
                    continue
                value = model.start[tags[0]]
                for i, (c, tag) in enumerate(zip(text, tags)):
                    value += model.emission[tag].get(c, model.unknown[tag])
                    if i:
                        value += model.transition[tags[i-1]][tag]
                candidates.append(value)
            words = model.segment(text)
            tags = [t for w in words for t in (['S'] if len(w) == 1 else ['B']+['M']*(len(w)-2)+['E'])]
            value = model.start[tags[0]]
            for i, (c, tag) in enumerate(zip(text, tags)):
                value += model.emission[tag].get(c, model.unknown[tag])
                if i:
                    value += model.transition[tags[i-1]][tag]
            self.assertAlmostEqual(value, max(candidates))
            self.assertEqual(''.join(words), text)

    def test_hmm_probabilities(self):
        model = HMMSegmenter().fit([['甲乙', '丙']])
        self.assertAlmostEqual(sum(math.exp(v) for v in model.start.values()), 1)
        for row in model.transition.values():
            self.assertAlmostEqual(sum(math.exp(v) for v in row.values()), 1)

    def test_bpe_against_naive_training(self):
        texts = ['甲乙甲乙，甲乙丙', '甲乙丙', '甲甲甲', 'abababa']
        corpus = Counter(tuple(c) for t in texts for c in chunks(t))
        expected = []
        for _ in range(6):
            counts = Counter()
            for seq, weight in corpus.items():
                for pair in zip(seq, seq[1:]):
                    counts[pair] += weight
            if not counts:
                break
            pair = min(counts, key=lambda p: (-counts[p], p))
            if counts[pair] < 2:
                break
            expected.append(pair)
            updated = Counter()
            for seq, weight in corpus.items():
                updated[merge(seq, pair)] += weight
            corpus = updated
        model = BPETokenizer().fit(texts, vocab_size=100, max_merges=6)
        self.assertEqual(model.merges, expected)
        text = '甲乙陌，ab🙂'
        self.assertEqual(''.join(model.tokenize(text)), text)

    def test_model_roundtrip(self):
        with tempfile.TemporaryDirectory(dir='Test1') as folder:
            for cls, train, method in [(HMMSegmenter, [['甲乙', '丙']], 'segment'), (BPETokenizer, ['甲乙甲乙丙'], 'tokenize')]:
                model = cls().fit(train)
                path = Path(folder)/'model.json'
                model.save(path)
                loaded = cls.load(path)
                self.assertEqual(getattr(model, method)('甲乙陌'), getattr(loaded, method)('甲乙陌'))

    def test_evaluation_zero_hits(self):
        test = [{'id': 'x', 'source_lines': [1], 'text': '甲乙丙', 'words': ['甲乙', '丙']}]
        metric, _, _ = evaluate_predictions(test, [['甲', '乙丙']], {'甲乙'}, 'test', 1)
        self.assertEqual(metric['f1'], 0)
        self.assertEqual(metric['oov_words'], 1)
        self.assertEqual(fertility(3, 2), 1.5)
        with self.assertRaises(ValueError):
            evaluate_predictions(test, [['甲乙丙', '']], set(), 'test', 1)

    def test_response_validation(self):
        self.assertEqual(parse_response('```json\n["甲","乙"]\n```', '甲乙'), ['甲', '乙'])
        for content in ['["甲"]', '[""]', '{}', '[1]', '解释文字']:
            with self.assertRaises((ValueError, TypeError)):
                parse_response(content, '甲乙')

    def test_api_mock_and_cache(self):
        import io
        response = {'choices': [{'message': {'content': '["甲", "乙"]'}}], 'usage': {'total_tokens': 5}}
        with tempfile.TemporaryDirectory(dir='Test1') as folder, patch.dict(os.environ, {
            'LLM_API_KEY': 'fake-test-key', 'LLM_MODEL': 'test-model', 'LLM_BASE_URL': 'https://example.invalid/v1'}):
            model = LLMSegmenter(cache_dir=folder, config_path=None)
            with patch('urllib.request.urlopen', return_value=io.BytesIO(json.dumps(response).encode())) as request:
                self.assertEqual(model.segment('甲乙'), ['甲', '乙'])
                self.assertFalse(model.last_info['cache_hit'])
                self.assertEqual(model.segment('甲乙'), ['甲', '乙'])
                self.assertTrue(model.last_info['cache_hit'])
                self.assertEqual(request.call_count, 1)
            for path in Path(folder).glob('*.json'):
                self.assertNotIn('fake-test-key', path.read_text(encoding='utf-8'))

    def test_matched_online_pipeline_without_network(self):
        from types import SimpleNamespace
        from main import run_llm_comparison
        train = [{'id': str(i), 'text': '甲乙丙丁戊己庚辛', 'words': list('甲乙丙丁戊己庚辛'), 'source_lines': [i+1]} for i in range(3)]
        test = [{'id': 'test'+str(i), 'text': t, 'words': list(t), 'source_lines': [10+i]} for i, t in enumerate(['陌生', '甲乙'])]
        class FakeClient:
            def __init__(self, examples, **kwargs):
                assert all(row['id'] in {r['id'] for r in train} for row in examples)
                self.model, self.base_url = 'mock', 'https://example.invalid/v1'
                self.last_info = {'cache_hit': False, 'request_seconds': 0.001}
            def segment(self, text):
                return list(text)
        with tempfile.TemporaryDirectory(dir='Test1') as folder, patch('main.LLMSegmenter', FakeClient):
            out = Path(folder)
            run_llm_comparison(SimpleNamespace(seed=2026, llm_limit=2, llm_temperature=None), train, test, {'甲'}, {'FMM': list}, out)
            metrics = json.loads((out/'llm_comparison/metrics.json').read_text(encoding='utf-8'))
            self.assertEqual([m['test_lines'] for m in metrics], [2, 2])
            self.assertEqual([m['f1'] for m in metrics], [1, 1])

    def test_deepseek_defaults_and_payload(self):
        import io
        response = {'choices': [{'message': {'content': '["甲"]'}}]}
        with tempfile.TemporaryDirectory(dir='Test1') as folder, patch.dict(os.environ, {'DEEPSEEK_API_KEY': 'fake-test-key'}, clear=True):
            model = LLMSegmenter(cache_dir=folder, config_path=None)
            self.assertEqual(model.model, 'deepseek-flash')
            self.assertEqual(model.base_url, 'https://api.deepseek.com')
            with patch('urllib.request.urlopen', return_value=io.BytesIO(json.dumps(response).encode())) as request:
                self.assertEqual(model.segment('甲'), ['甲'])
                sent = request.call_args.args[0]
                self.assertEqual(sent.full_url, 'https://api.deepseek.com/chat/completions')
                self.assertEqual(json.loads(sent.data)['thinking'], {'type': 'disabled'})

    def test_project_dotenv(self):
        from config import load_settings
        with tempfile.TemporaryDirectory(dir='Test1') as folder, patch.dict(os.environ, {'LLM_MODEL': 'old', 'OTHER': 'fallback'}, clear=True):
            path = Path(folder)/'.env'
            path.write_text('# comment\nexport LLM_MODEL="deepseek-flash" # comment\nLLM_API_KEY=sample#key\nLLM_BASE_URL=https://api.deepseek.com # comment\n', encoding='utf-8-sig')
            settings = load_settings(path)
            self.assertEqual(settings['LLM_MODEL'], 'deepseek-flash')
            self.assertEqual(settings['OTHER'], 'fallback')
            model = LLMSegmenter(cache_dir=Path(folder)/'cache', config_path=path)
            self.assertEqual(model.api_key, 'sample#key')
            self.assertEqual(model.base_url, 'https://api.deepseek.com')
            self.assertEqual(os.environ['LLM_MODEL'], 'old')
            path.write_text('invalid line', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, '第 1 行'):
                load_settings(path)

    def test_retry_feedback_names_changed_character(self):
        import io
        responses = [{'choices': [{'message': {'content': '["必定"]'}}]}, {'choices': [{'message': {'content': '["必将"]'}}]}]
        with tempfile.TemporaryDirectory(dir='Test1') as folder, patch.dict(os.environ, {'LLM_API_KEY': 'fake', 'LLM_MODEL': 'test', 'LLM_BASE_URL': 'https://example.invalid'}, clear=True):
            model = LLMSegmenter(cache_dir=folder, config_path=None)
            with patch('urllib.request.urlopen', side_effect=[io.BytesIO(json.dumps(r).encode()) for r in responses]) as request, patch('time.sleep'):
                self.assertEqual(model.segment('必将'), ['必将'])
                correction = json.loads(request.call_args.args[0].data)['messages'][-1]['content']
                self.assertIn("应为 '将'", correction)
                self.assertIn("返回 '定'", correction)
                self.assertEqual(model.last_info['attempts'], 2)


if __name__ == '__main__':
    unittest.main(verbosity=2)
