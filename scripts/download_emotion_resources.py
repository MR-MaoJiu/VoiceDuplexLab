"""下载官方RAVDESS情绪语句补充包，支持断点续传与完整校验。"""
import argparse
import hashlib
from pathlib import Path
import time

import requests

from voice_lab.download import atomic_json

NAME = 'Audio_Speech_Actors_01-24.zip'
URL = 'https://zenodo.org/api/records/1188976/files/' + NAME + '/content'
SIZE = 208468073
MD5 = 'bc696df654c87fed845eb13823edef8a'


def valid(path):
    if not path.is_file() or path.stat().st_size != SIZE:
        return False
    digest = hashlib.md5()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest() == MD5


def download(directory):
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / NAME
    partial = directory / (NAME + '.partial')
    for attempt in range(4):
        if valid(target):
            break
        if valid(partial):
            partial.replace(target)
            break
        if partial.exists() and partial.stat().st_size >= SIZE:
            partial.unlink()
        offset = partial.stat().st_size if partial.exists() else 0
        try:
            with requests.get(URL, headers={'Range': f'bytes={offset}-'} if offset else {},
                              stream=True, timeout=(20, 90)) as response:
                response.raise_for_status()
                append = offset > 0 and response.status_code == 206
                if append and not response.headers.get('Content-Range', '').startswith(f'bytes {offset}-'):
                    raise ValueError('服务端返回的续传位置不符')
                with partial.open('ab' if append else 'wb') as stream:
                    for chunk in response.iter_content(1024 * 1024):
                        stream.write(chunk)
            if not valid(partial):
                raise ValueError('录音包校验失败')
            partial.replace(target)
            break
        except (requests.RequestException, ValueError) as exc:
            if attempt == 3:
                raise
            delay = 20 if getattr(getattr(exc, 'response', None), 'status_code', None) == 429 else 2
            print(f'下载遇到{type(exc).__name__}，{delay}秒后重试', flush=True)
            time.sleep(delay)
    digest = hashlib.sha256()
    with target.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    atomic_json(directory / 'provenance.json', {
        'source': 'https://zenodo.org/records/1188976', 'file': NAME,
        'bytes': SIZE, 'md5': MD5, 'sha256': digest.hexdigest(),
        'license': 'CC BY-NC-SA 4.0', 'language': '英语',
        'emotions': ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgusted', 'surprised'],
        'scope': '完整情绪语句补充素材，下载后仍需构建对齐的联合训练清单与中文验收集',
    })
    print(f'情绪语句素材已下载并校验 {target}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('data/emotion/RAVDESS'))
    download(parser.parse_args().output)
