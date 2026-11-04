"""Run two Shake-Shake + AutoAugment configurations after an active experiment."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from run import launch
from train import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', required=True)
    parser.add_argument('--after', required=True)
    args = parser.parse_args()
    out = ROOT / 'runs' / args.name
    out.mkdir(parents=True, exist_ok=False)
    options = ['--architecture', 'shake', '--depth', '26', '--autoaugment', '--cutout', '16',
               '--lr', '0.2', '--weight-decay', '0.0001', '--batch-size', '128']
    config = {'after': args.after, 'widths': [32, 40], 'epochs': 1800, 'options': options,
              'commit': subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()}
    (out / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    def status(phase):
        (out / 'status.json').write_text(json.dumps({'phase': phase}) + '\n')
        print(phase, flush=True)
    status('waiting_for_' + args.after)
    prior = ROOT / 'runs' / args.after
    receipt = json.loads((ROOT / 'runs' / (args.after + '.launch.json')).read_text())
    while True:
        selection = prior / 'final_selection.json'
        if selection.exists():
            try:
                if json.loads(selection.read_text())['status'] == 'complete':
                    break
            except json.JSONDecodeError:
                pass
        process = Path('/proc') / str(receipt['pid'])
        try:
            command = process.joinpath('cmdline').read_bytes().decode().rstrip('\0').split('\0')
        except FileNotFoundError:
            raise RuntimeError('Previous experiment exited before completing its final selection')
        if command != receipt['command']:
            raise RuntimeError('Previous experiment is no longer running under its recorded PID')
        time.sleep(10)
    status('training')
    jobs = []
    for device, width in enumerate(config['widths']):
        name = f'{args.name}_w{width}'
        jobs.append((name, launch(name, options + ['--width', str(width)], device, epochs=config['epochs'])))
    failures = [name for name, process in jobs if process is not None and process.wait() != 0]
    if failures:
        raise RuntimeError(f'Shake-Shake training failed: {failures}')
    results = []
    for name, _ in jobs:
        result = json.loads((ROOT / 'runs' / name / 'result.json').read_text())
        if result['status'] != 'complete' or result['plain']['total'] != 5000 or result['flip']['total'] != 5000:
            raise ValueError(f'Incomplete validation: {name}')
        results.append({'name': name, **result})
    (out / 'result.json').write_text(json.dumps({'status': 'complete', 'finalists': results}, indent=2) + '\n')
    references = ['structure_search_20261029_trial11_final', 'wrn16_randaugment_mix']
    references += [result['name'] for result in json.loads((prior / 'result.json').read_text())['finalists']]
    status('preparing_local_prediction')
    subprocess.run([sys.executable, str(ROOT / 'finish_search.py'), args.name,
                    '--search-pid', str(os.getpid()), '--reference', *references], check=True)
    status('complete')


if __name__ == '__main__':
    main()
