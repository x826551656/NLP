"""项目统一配置：NLP 根目录 .env 优先，未配置项回退到进程环境变量。"""
import os
import re
from pathlib import Path


def load_settings(env_file=Path('.env')):
    settings = dict(os.environ)
    if env_file is None:
        return settings
    path = Path(env_file)
    if not path.is_file():
        return settings
    for number, original in enumerate(path.read_text(encoding='utf-8-sig').splitlines(), 1):
        line = original.strip()
        if not line or line.startswith('#'):
            continue
        if line.startswith('export '):
            line = line[7:].lstrip()
        key, separator, value = line.partition('=')
        key, value = key.strip(), value.strip()
        if not separator or not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', key):
            raise ValueError(f'.env 第 {number} 行格式错误；需要 KEY=VALUE')
        if value.startswith(('"', "'")):
            quote = value[0]
            end = value.find(quote, 1)
            if end < 0 or (value[end+1:].strip() and not value[end+1:].lstrip().startswith('#')):
                raise ValueError(f'.env 第 {number} 行引号格式错误')
            value = value[1:end]
        else:
            value = re.split(r'\s+#', value, maxsplit=1)[0].rstrip()
        settings[key] = value
    return settings
