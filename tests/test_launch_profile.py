"""启动入口应选择已验收的适配器配置，并尊重显式覆盖。"""
import os
from pathlib import Path
import shutil
import subprocess
import pytest


@pytest.mark.parametrize('script',['serve.sh','backend.sh'])
def test_default_launcher_uses_ready_profile_and_preserves_override(tmp_path,script):
    root=Path(__file__).resolve().parents[1]
    (tmp_path/'scripts').mkdir()
    (tmp_path/'configs').mkdir()
    (tmp_path/'.venv/bin').mkdir(parents=True)
    for name in ('common.sh',script):shutil.copy2(root/'scripts'/name,tmp_path/'scripts'/name)
    for name in ('default.yaml','repair-ready.yaml','bundle-runtime.yaml'):(tmp_path/'configs'/name).write_text('{}')
    fake=tmp_path/'.venv/bin/python'
    fake.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$VOICE_CONFIG"\n')
    fake.chmod(0o755)
    env={**os.environ,'BACKEND_ENGINE':'transformers'}
    env.pop('VOICE_CONFIG',None)
    command=['bash',str(tmp_path/'scripts'/script)]
    result=subprocess.run(command,env=env,text=True,capture_output=True,check=True)
    assert result.stdout.strip()==str(tmp_path/'configs/repair-ready.yaml')
    env['VOICE_CONFIG']=str(tmp_path/'configs/custom.yaml')
    result=subprocess.run(command,env=env,text=True,capture_output=True,check=True)
    assert result.stdout.strip()==env['VOICE_CONFIG']
