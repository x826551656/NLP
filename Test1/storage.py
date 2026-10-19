"""UTF-8 JSON / JSONL 输出：先写同目录临时文件，再替换完成文件。"""
import json
import os
import tempfile
from pathlib import Path


def _write(path, chunks):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                         prefix=path.name+'.', suffix='.tmp', delete=False) as f:
            temporary = Path(f.name)
            for chunk in chunks:
                f.write(chunk)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def save(path, obj):
    _write(path, [json.dumps(obj, ensure_ascii=False, indent=2)])


def jsonl(path, rows):
    _write(path, (json.dumps(row, ensure_ascii=False)+'\n' for row in rows))
