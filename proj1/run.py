"""Run two candidates, then retrain the validation winner on all training data."""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CANDIDATES = [
    ('wrn28_cutout', ['--depth', '28', '--width', '58']),
    ('wrn16_randaugment_mix', ['--depth', '16', '--width', '84', '--randaugment', '6',
                              '--cutout', '0', '--mixup', '0.2', '--cutmix', '1']),
]


def select_result(results):
    choices = []
    for name, result in results.items():
        if result['status'] != 'complete' or result['plain']['total'] != 5000 or result['flip']['total'] != 5000:
            raise ValueError(f'Incomplete validation result for {name}')
        mode = 'flip' if result['flip']['accuracy'] > result['plain']['accuracy'] else 'plain'
        choices.append({'name': name, 'inference': mode, 'validation_accuracy': result[mode]['accuracy']})
    return max(choices, key=lambda item: item['validation_accuracy'])


def launch(name, options, device, epochs=300):
    directory = ROOT / 'runs' / name
    command = [sys.executable, '-u', str(ROOT / 'train.py'), '--name', name,
               '--device', f'cuda:{device}', '--epochs', str(epochs), *options]
    if (directory / 'result.json').exists():
        return None
    if (directory / 'last.pt').exists():
        command.append('--resume')
    with (ROOT / 'runs' / f'{name}.log').open('a') as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
    print(json.dumps({'name': name, 'pid': process.pid, 'command': command}), flush=True)
    return process


def main():
    (ROOT / 'runs').mkdir(exist_ok=True)
    processes = [(name, launch(name, options, device)) for device, (name, options) in enumerate(CANDIDATES)]
    failures = []
    for name, process in processes:
        if process is not None and process.wait() != 0:
            failures.append(name)
    if failures:
        raise RuntimeError(f'Training failed; inspect logs for {failures}')
    results = {name: json.loads((ROOT / 'runs' / name / 'result.json').read_text()) for name, _ in CANDIDATES}
    winner = select_result(results)
    winner['final_run'] = winner['name'] + '_full'
    (ROOT / 'runs' / 'selection.json').write_text(json.dumps(winner, indent=2) + '\n')
    print(json.dumps(winner), flush=True)
    options = next(options for name, options in CANDIDATES if name == winner['name'])
    final = launch(winner['final_run'], options + ['--full-data'], 0)
    if final is not None and final.wait() != 0:
        raise RuntimeError('Final training failed; inspect its log')
    print('COMPLETE: final weights at runs/' + winner['final_run'] + '/final.pt', flush=True)


if __name__ == '__main__':
    main()
