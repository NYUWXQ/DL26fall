"""Compare SE-ResNet/RICAP and heterogeneous-teacher distillation locally."""
import argparse
import json
import os
import subprocess
import sys
import time

from run import launch
from train import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', required=True)
    args = parser.parse_args()
    out = ROOT / 'runs' / args.name
    out.mkdir(parents=True, exist_ok=False)
    teacher = ROOT / 'runs/distill_20261030_teacher256/best.pt'
    shake = ROOT / 'runs/shake_aa_20261030_w40/best.pt'
    reference = 'distill_20261030_frontloaded_t2_e600'
    specifications = [
        (args.name + '_se164_ricap', 1800,
         ['--architecture', 'se', '--depth', '164', '--width', '16',
          '--autoaugment', '--ricap', '0.3', '--random-erasing', '0.5', '--cutout', '0',
          '--lr', '0.1', '--weight-decay', '0.0001', '--warmup-epochs', '60', '--min-lr', '0.0001']),
        (args.name + '_multiteacher', 600,
         ['--stage-widths', '136', '192', '232', '--stage-blocks', '3', '3', '2',
          '--lr', '0.0816831165758183', '--weight-decay', '0.00023238214295871968',
          '--randaugment', '9', '--cutout', '0', '--cutmix', '1',
          '--teacher-checkpoint', str(teacher), '--extra-teacher-checkpoint', str(shake),
          '--distill-weight', '0.75', '--temperature', '2']),
    ]
    for path in (teacher, shake, ROOT / 'runs' / reference / 'result.json'):
        if not path.is_file():
            raise FileNotFoundError(path)
    commit = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    config = {'commit': commit, 'runs': specifications, 'reference': reference,
              'training_data': 'CIFAR-10 official train split; fixed 45000/5000 split',
              'selection': 'maximum plain or flip validation accuracy; retain reference on ties',
              'teacher_aggregation': 'equal probabilities at temperature 2',
              'external_actions': False}
    (out / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    jobs = []
    for device, (name, epochs, options) in enumerate(specifications):
        process = launch(name, options, device, epochs)
        receipt = {'name': name, 'pid': process.pid, 'command': process.args, 'device': device,
                   'commit': commit, 'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
        (ROOT / 'runs' / (name + '.launch.json')).write_text(json.dumps(receipt, indent=2) + '\n')
        jobs.append((name, process))
    failures = [name for name, process in jobs if process.wait() != 0]
    if failures:
        raise RuntimeError(f'Training failed: {failures}')
    results = []
    for name, _ in jobs:
        result = json.loads((ROOT / 'runs' / name / 'result.json').read_text())
        if result['status'] != 'complete' or result['plain']['total'] != 5000 or result['flip']['total'] != 5000:
            raise ValueError(f'Incomplete validation: {name}')
        results.append({'name': name, **result})
    (out / 'result.json').write_text(json.dumps({'status': 'complete', 'finalists': results}, indent=2) + '\n')
    subprocess.run([sys.executable, '-u', str(ROOT / 'finish_search.py'), args.name,
                    '--search-pid', str(os.getpid()), '--device', 'cuda:1',
                    '--precision', 'fp32',
                    '--reference', reference], check=True)


if __name__ == '__main__':
    main()
