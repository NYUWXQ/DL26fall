"""Retrain a validation winner on all training data and write a local CSV."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from train import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('search', help='Search run name')
    parser.add_argument('--search-pid', type=int, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--precision', choices=['bf16', 'fp32'], default='bf16')
    parser.add_argument('--reference', nargs='+', required=True)
    args = parser.parse_args()
    directory = ROOT / 'runs' / args.search
    result_file = directory / 'result.json'
    while not result_file.exists():
        os.kill(args.search_pid, 0)
        time.sleep(10)
    search = json.loads(result_file.read_text())
    if search['status'] != 'complete':
        raise ValueError('Search did not complete')
    candidates = []
    for name in args.reference + [item['name'] for item in search['finalists']]:
        result = json.loads((ROOT / 'runs' / name / 'result.json').read_text())
        if result['status'] != 'complete':
            raise ValueError(f'Incomplete run: {name}')
        score = max(result['plain']['accuracy'], result['flip']['accuracy'])
        candidates.append({'name': name, 'accuracy': score})
    # References come first, so a tie retains the existing model.
    winner = max(candidates, key=lambda item: item['accuracy'])
    selection = {'winner': winner, 'candidates': candidates, 'status': 'selected'}
    selection_file = directory / 'final_selection.json'
    selection_file.write_text(json.dumps(selection, indent=2) + '\n')
    reference = max(candidates[:len(args.reference)], key=lambda item: item['accuracy'])
    if winner['accuracy'] <= reference['accuracy']:
        selection.update(status='complete', action='retain existing full-data model')
        selection_file.write_text(json.dumps(selection, indent=2) + '\n')
        print(json.dumps(selection), flush=True)
        return
    config = json.loads((ROOT / 'runs' / winner['name'] / 'config.json').read_text())
    name = winner['name'] + '_full'
    options = []
    for key in ('architecture', 'depth', 'width', 'dropout', 'epochs', 'batch_size', 'lr', 'weight_decay',
                'randaugment', 'cutout', 'mixup', 'cutmix', 'workers', 'groups',
                'eval_batch_size', 'distill_weight', 'temperature', 'ricap', 'random_erasing',
                'warmup_epochs', 'min_lr'):
        if key in config:
            options += ['--' + key.replace('_', '-'), str(config[key])]
    for key in ('stage_widths', 'stage_blocks'):
        if config.get(key) is not None:
            options += ['--' + key.replace('_', '-'), *map(str, config[key])]
    if config.get('teacher_checkpoint'):
        options += ['--teacher-checkpoint', config['teacher_checkpoint']]
    for checkpoint in config.get('extra_teacher_checkpoint', []):
        options += ['--extra-teacher-checkpoint', checkpoint]
    if config.get('init_checkpoint'):
        options += ['--init-checkpoint', config['init_checkpoint']]
    if config.get('autoaugment'):
        options.append('--autoaugment')
    def run(script, *options):
        command = [sys.executable, '-u', str(ROOT / script), *map(str, options)]
        print(json.dumps({'command': command}), flush=True)
        subprocess.run(command, check=True)
    completed = ROOT / 'runs' / name / 'result.json'
    if completed.exists():
        if json.loads(completed.read_text())['status'] != 'complete':
            raise ValueError(f'Full-data run is incomplete: {name}')
        print(json.dumps({'reuse_full_run': name}), flush=True)
    else:
        run('train.py', '--name', name, '--device', args.device, '--full-data', *options)
    recipe = directory / 'winner_recipe'
    run('refine.py', ROOT / 'runs' / winner['name'] / 'best.pt', '--out', recipe,
        '--device', args.device, '--precision', args.precision)
    checkpoint = directory / 'winner_full.pt'
    run('export.py', ROOT / 'runs' / name / 'last.pt', '--recipe', recipe,
        '--out', checkpoint, '--device', args.device)
    csv = directory / 'submission' / 'submission.csv'
    run('predict.py', checkpoint, '--out', csv, '--device', args.device)
    selection.update(status='complete', full_run=name, checkpoint=str(checkpoint), csv=str(csv))
    selection_file.write_text(json.dumps(selection, indent=2) + '\n')
    print(json.dumps(selection), flush=True)


if __name__ == '__main__':
    main()
