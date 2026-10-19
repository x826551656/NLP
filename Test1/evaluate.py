"""词级完整边界 micro P/R/F1、OOV 召回与子词 Fertility。"""


def spans(words):
    start = 0
    result = set()
    for word in words:
        result.add((start, start + len(word)))
        start += len(word)
    return result


def fertility(token_count, original_word_count):
    """BPE token 总数 / 同一批文本的原始标准词总数。"""
    if token_count < 0 or original_word_count <= 0:
        raise ValueError('token_count 必须非负，原始词数必须为正')
    return token_count / original_word_count


def evaluate_predictions(test, predictions, vocab, name, elapsed):
    if len(test) != len(predictions) or not test:
        raise ValueError('预测和测试集必须非空且逐条对应')
    total_gold = sum(len(row['words']) for row in test)
    oov_total = sum(word not in vocab for row in test for word in row['words'])
    correct = predicted = exact = oov_correct = 0
    errors, records = [], []
    for row, pred in zip(test, predictions):
        assert ''.join(pred) == row['text']
        gold_spans, pred_spans = spans(row['words']), spans(pred)
        correct += len(gold_spans & pred_spans)
        predicted += len(pred)
        exact += pred == row['words']
        offset = 0
        oov = []
        for word in row['words']:
            if word not in vocab:
                oov.append(word)
                oov_correct += (offset, offset + len(word)) in pred_spans
            offset += len(word)
        record = {'id': row['id'], 'source_lines': row['source_lines'],
                  'text': row['text'], 'gold': row['words'], 'predicted': pred,
                  'oov_words': oov, 'boundary_errors': len(gold_spans ^ pred_spans)}
        records.append(record)
        if pred != row['words']:
            errors.append(record)
    p, r = correct / predicted, correct / total_gold
    metric = {'method': name, 'precision': p, 'recall': r, 'f1': 2*p*r/(p+r),
              'correct_words': correct, 'predicted_words': predicted,
              'gold_words': total_gold, 'exact_lines': exact,
              'test_lines': len(test), 'oov_words': oov_total,
              'oov_correct': oov_correct, 'oov_recall': oov_correct/oov_total if oov_total else None,
              'segmentation_seconds': elapsed,
              'characters_per_second': sum(len(r['text']) for r in test)/elapsed}
    return metric, records, errors


def typical_errors(errors, count=10):
    selected = []
    for has_oov in [False, True]:
        subset = sorted((e for e in errors if bool(e['oov_words']) == has_oov),
                        key=lambda e: (len(e['text']), e['id']))
        selected.extend(subset[:count])
    return selected
