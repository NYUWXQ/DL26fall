"""Wait for the current local experiments and prepare the batch's winning model."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from train import ROOT


def wait_for(name, filename):
    directory = ROOT / 'runs'
    receipt = json.loads((directory / (name + '.launch.json')).read_text())
    while True:
        path = directory / name / filename
        if path.exists():
            try:
                result = json.loads(path.read_text())
                if result['status'] == 'complete':
                    return result
            except json.JSONDecodeError:
                pass
        command = Path('/proc', str(receipt['pid']), 'cmdline')
        if not command.exists() or command.read_bytes().decode().rstrip('\0').split('\0') != receipt['command']:
            raise RuntimeError(f'{name} exited before completing {filename}')
        time.sleep(30)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', required=True)
    args = parser.parse_args()
    out = ROOT / 'runs' / args.name
    out.mkdir(parents=True, exist_ok=False)
    commit = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    (out / 'config.json').write_text(json.dumps({'commit': commit}, indent=2) + '\n')
    kd_name = 'distill_20261030_frontloaded_t2_e600'
    kd = wait_for(kd_name, 'result.json')
    wait_for('lejepa_20261030', 'final_selection.json')
    (out / 'result.json').write_text(json.dumps(
        {'status': 'complete', 'finalists': [{'name': kd_name, **kd}]}, indent=2) + '\n')
    subprocess.run([sys.executable, '-u', str(ROOT / 'finish_search.py'), args.name,
                    '--search-pid', str(os.getpid()), '--device', 'cuda:1',
                    '--reference', 'distill_20261030_frontloaded',
                    'structure_search_20261029_trial11_final', 'wrn16_randaugment_mix',
                    'distill_20261030_wrn16', 'shake_aa_20261030_w32',
                    'shake_aa_20261030_w40', 'lejepa_20261030_finetune'], check=True)
    selected = json.loads((out / 'final_selection.json').read_text())
    for parent in (args.name, 'lejepa_20261030', 'distill_20261030'):
        candidate = json.loads((ROOT / 'runs' / parent / 'final_selection.json').read_text())
        if candidate['winner']['name'] == selected['winner']['name'] and candidate.get('csv'):
            selected.update(checkpoint=candidate['checkpoint'], csv=candidate['csv'], full_run=candidate['full_run'])
            break
    if not selected.get('csv') or not Path(selected['csv']).is_file() or not Path(selected['checkpoint']).is_file():
        raise ValueError('Winning full-data checkpoint or CSV is missing')
    (out / 'delivery.json').write_text(json.dumps(selected, indent=2) + '\n')
    print(json.dumps(selected), flush=True)


if __name__ == '__main__':
    main()
