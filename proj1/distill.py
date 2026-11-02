"""Train two larger teachers, compare distilled students, and prepare local predictions."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from run import launch
from train import ROOT


def run_pair(specifications):
    jobs = [(name, launch(name, options, device))
            for device, (name, options) in enumerate(specifications)]
    failures = []
    for name, process in jobs:
        if process is not None and process.wait() != 0:
            failures.append(name)
    if failures:
        raise RuntimeError(f'Failed runs: {failures}')
    results = []
    for name, _ in jobs:
        result = json.loads((ROOT / 'runs' / name / 'result.json').read_text())
        if result['status'] != 'complete':
            raise ValueError(f'Incomplete result: {name}')
        results.append({'name': name, **result})
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', required=True)
    args = parser.parse_args()
    out = ROOT / 'runs' / args.name
    out.mkdir(parents=True, exist_ok=False)
    shared = ['--randaugment', '6', '--cutout', '0', '--mixup', '0.2', '--cutmix', '1']
    specifications = [(f'{args.name}_teacher{width}',
                       ['--teacher-only', '--depth', '28', '--width', str(width),
                        '--dropout', '0.1', '--eval-batch-size', '128', *shared])
                      for width in (160, 256)]
    config = {'teachers': specifications, 'seed': 42, 'epochs': 300,
              'split': 'existing stratified 45000/5000 CIFAR-10 training split',
              'commit': subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()}
    (out / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    teachers = run_pair(specifications)
    teacher = max(teachers, key=lambda result: max(result['plain']['accuracy'], result['flip']['accuracy']))
    checkpoint = ROOT / 'runs' / teacher['name'] / 'best.pt'
    (out / 'teacher_selection.json').write_text(json.dumps({'selected': teacher['name'], 'teachers': teachers}, indent=2) + '\n')
    kd = ['--teacher-checkpoint', str(checkpoint), '--distill-weight', '0.5', '--temperature', '4']
    students = [
        (args.name + '_frontloaded', ['--stage-widths', '136', '192', '232', '--stage-blocks', '3', '3', '2',
                                     '--lr', '0.0816831165758183', '--weight-decay', '0.00023238214295871968',
                                     '--randaugment', '9', '--cutout', '0', '--mixup', '0', '--cutmix', '1', *kd]),
        (args.name + '_wrn16', ['--depth', '16', '--width', '84', *shared, *kd]),
    ]
    results = run_pair(students)
    summary = {'status': 'complete', 'teachers': teachers, 'selected_teacher': teacher['name'], 'finalists': results}
    (out / 'result.json').write_text(json.dumps(summary, indent=2) + '\n')
    subprocess.run([sys.executable, str(ROOT / 'finish_search.py'), args.name, '--search-pid', str(os.getpid()),
                    '--reference', 'structure_search_20261029_trial11_final', 'wrn16_randaugment_mix'], check=True)
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
