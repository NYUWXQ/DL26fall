"""Pretrain LeJEPA, fine-tune its backbone, and prepare local predictions if better."""
import argparse
import json
import os
import subprocess
import sys

from run import launch
from train import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', required=True)
    parser.add_argument('--device', type=int, default=0)
    args = parser.parse_args()
    out = ROOT / 'runs' / args.name
    out.mkdir(parents=True, exist_ok=False)
    pretrain_name = args.name + '_pretrain'
    student_name = args.name + '_finetune'
    references = ['distill_20261030_frontloaded', 'distill_20261030_wrn16',
                  'structure_search_20261029_trial11_final', 'wrn16_randaugment_mix',
                  'shake_aa_20261030_w32', 'shake_aa_20261030_w40']
    config = {'pretrain': pretrain_name, 'student': student_name, 'references': references,
              'device': args.device, 'commit': subprocess.check_output(
                  ['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()}
    (out / 'config.json').write_text(json.dumps(config, indent=2) + '\n')

    def status(phase):
        (out / 'status.json').write_text(json.dumps({'phase': phase}) + '\n')
        print(phase, flush=True)

    def run(script, *options):
        command = [sys.executable, '-u', str(ROOT / script), *map(str, options)]
        print(json.dumps({'command': command}), flush=True)
        subprocess.run(command, check=True)

    status('pretraining')
    run('pretrain_lejepa.py', '--name', pretrain_name, '--device', f'cuda:{args.device}')
    result = json.loads((ROOT / 'runs' / pretrain_name / 'result.json').read_text())
    if result['status'] != 'complete':
        raise ValueError('LeJEPA pretraining did not complete')
    status('fine_tuning')
    options = ['--stage-widths', '136', '192', '232', '--stage-blocks', '3', '3', '2',
               '--lr', '0.0816831165758183', '--weight-decay', '0.00023238214295871968',
               '--randaugment', '9', '--cutout', '0', '--mixup', '0', '--cutmix', '1',
               '--init-checkpoint', str(ROOT / 'runs' / pretrain_name / 'backbone.pt')]
    process = launch(student_name, options, args.device)
    if process is not None and process.wait() != 0:
        raise RuntimeError('LeJEPA fine-tuning failed; inspect its log')
    result = json.loads((ROOT / 'runs' / student_name / 'result.json').read_text())
    if result['status'] != 'complete' or result['plain']['total'] != 5000 or result['flip']['total'] != 5000:
        raise ValueError('Incomplete LeJEPA validation')
    (out / 'result.json').write_text(json.dumps(
        {'status': 'complete', 'finalists': [{'name': student_name, **result}]}, indent=2) + '\n')
    status('preparing_local_prediction')
    run('finish_search.py', args.name, '--search-pid', os.getpid(),
        '--device', f'cuda:{args.device}', '--reference', *references)
    status('complete')


if __name__ == '__main__':
    main()
