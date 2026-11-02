"""Exercise the search scheduler with short subprocesses instead of GPU jobs."""
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch

import search


def main():
    fake_train = '''import argparse, json, time
from pathlib import Path
p = argparse.ArgumentParser()
p.add_argument('--name')
p.add_argument('--epochs', type=int)
a, _ = p.parse_known_args()
out = Path(__file__).parent / 'runs' / a.name
out.mkdir(parents=True, exist_ok=False)
with (out / 'history.jsonl').open('w') as f:
    f.write(json.dumps({'epoch': 40, 'best_accuracy': 0.8}) + '\\n')
time.sleep(0.1)
(out / 'result.json').write_text(json.dumps({'status': 'complete',
    'plain': {'accuracy': 0.8, 'total': 5000},
    'flip': {'accuracy': 0.81, 'total': 5000}}))
'''
    sleep = search.time.sleep
    with TemporaryDirectory(dir=search.ROOT / 'runs', prefix='search_check_') as directory:
        root = Path(directory)
        (root / 'train.py').write_text(fake_train)
        argv = ['search.py', '--name', 'check', '--trials', '7', '--epochs', '40',
                '--finalists', '2', '--final-epochs', '2']
        with patch.object(search, 'ROOT', root), patch.object(sys, 'argv', argv), \
             patch.object(search.time, 'sleep', lambda seconds: sleep(0.05)):
            search.main()
        result = json.loads((root / 'runs/check/result.json').read_text())
        assert len(result['trials']) == 7
        assert all(trial['state'] == 'COMPLETE' for trial in result['trials'])
        assert len(result['finalists']) == 2
        receipts = list((root / 'runs/check').glob('*.launch.json'))
        assert len(receipts) == 9
        assert {json.loads(path.read_text())['device'] for path in receipts} == {0, 1}
    print('PASS: search queue, subprocess results, and finalist retraining')


if __name__ == '__main__':
    main()
