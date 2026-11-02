"""Search residual-network shapes and training settings on the fixed holdout."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import optuna

from train import ROOT

ARCHITECTURES = {
    'backloaded': ([56, 112, 296], [1, 2, 3], 1),
    'balanced': ([72, 144, 264], [2, 3, 3], 1),
    'frontloaded': ([136, 192, 232], [3, 3, 2], 1),
    'latewide': ([56, 112, 376], [2, 2, 2], 1),
    'group4': ([88, 176, 352], [2, 3, 3], 4),
    'group8': ([120, 240, 480], [2, 2, 2], 8),
    'group4deep': ([72, 144, 304], [2, 4, 4], 4),
}


def training_options(trial):
    shape = trial.suggest_categorical('architecture', list(ARCHITECTURES))
    widths, blocks, groups = ARCHITECTURES[shape]
    return [
        '--stage-widths', *map(str, widths), '--stage-blocks', *map(str, blocks),
        '--groups', str(groups),
        '--lr', str(trial.suggest_float('lr', 0.04, 0.2, log=True)),
        '--weight-decay', str(trial.suggest_float('weight_decay', 0.0001, 0.001, log=True)),
        '--dropout', str(trial.suggest_categorical('dropout', [0.0, 0.1, 0.2])),
        '--randaugment', str(trial.suggest_int('randaugment', 4, 10)),
        '--mixup', str(trial.suggest_categorical('mixup', [0.0, 0.2, 0.4])),
        '--cutmix', str(trial.suggest_categorical('cutmix', [0.0, 1.0])), '--cutout', '0',
    ]


def launch(name, options, device, epochs, out):
    command = [sys.executable, '-u', str(ROOT / 'train.py'), '--name', name,
               '--device', f'cuda:{device}', '--epochs', str(epochs), *options]
    with (out / f'{name}.log').open('x') as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    record = {'name': name, 'pid': process.pid, 'device': device, 'command': command}
    (out / f'{name}.launch.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record), flush=True)
    return process


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', required=True)
    parser.add_argument('--devices', nargs='+', type=int, default=[0, 1])
    parser.add_argument('--trials', type=int, default=12)
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--finalists', type=int, default=2)
    parser.add_argument('--final-epochs', type=int, default=300)
    parser.add_argument('--after', nargs='*', default=[])
    parser.add_argument('--after-pids', nargs='*', type=int, default=[])
    args = parser.parse_args()
    if args.trials < len(ARCHITECTURES) or args.epochs < 40 or args.finalists < 1:
        parser.error('Use at least seven trials, 40 epochs, and one finalist')
    if len(set(args.devices)) != len(args.devices):
        parser.error('Each GPU must appear once')
    if args.after and (len(args.after) != len(args.devices) or len(args.after_pids) != len(args.devices)):
        parser.error('Supply one preceding run and PID per GPU')
    out = ROOT / 'runs' / args.name
    out.mkdir(parents=True, exist_ok=False)
    config = vars(args) | {'architectures': ARCHITECTURES, 'seed': 42,
                           'selection': 'fixed 5000-image holdout; top short runs retrained from scratch',
                           'commit': subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()}
    (out / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    study = optuna.create_study(
        study_name=args.name, direction='maximize', storage=f'sqlite:///{out / "study.db"}',
        sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=7),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=4, n_warmup_steps=40, interval_steps=10),
    )
    for architecture in ARCHITECTURES:
        study.enqueue_trial({'architecture': architecture, 'lr': 0.1, 'weight_decay': 0.0005,
                             'dropout': 0.0, 'randaugment': 6, 'mixup': 0.2, 'cutmix': 1.0})
    active, started = {}, 0
    try:
        while started < args.trials or active:
            for index, device in enumerate(args.devices):
                if device in active or started == args.trials:
                    continue
                if args.after and not (ROOT / 'runs' / args.after[index] / 'result.json').exists():
                    os.kill(args.after_pids[index], 0)
                    continue
                trial = study.ask()
                options = training_options(trial)
                name = f'{args.name}_trial{trial.number:02d}'
                trial.set_user_attr('options', options)
                trial.set_user_attr('run_name', name)
                process = launch(name, options, device, args.epochs, out)
                active[device] = (trial, process, name, 0)
                started += 1
            for device, (trial, process, name, reported) in list(active.items()):
                directory = ROOT / 'runs' / name
                code = process.poll()
                if code is not None:
                    if code != 0:
                        study.tell(trial, state=optuna.trial.TrialState.FAIL)
                        raise RuntimeError(f'{name} exited {code}; inspect its log')
                    result = json.loads((directory / 'result.json').read_text())
                    value = max(result['plain']['accuracy'], result['flip']['accuracy'])
                    study.tell(trial, value)
                    print(json.dumps({'trial': trial.number, 'state': 'complete', 'accuracy': value}), flush=True)
                    del active[device]
                    continue
                history = directory / 'history.jsonl'
                if not history.exists():
                    continue
                lines = history.read_text().splitlines()
                if not lines:
                    continue
                try:
                    row = json.loads(lines[-1])
                except json.JSONDecodeError:
                    continue
                if row['epoch'] <= reported:
                    continue
                trial.report(row['best_accuracy'], row['epoch'])
                active[device] = (trial, process, name, row['epoch'])
                if trial.should_prune():
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait()
                    study.tell(trial, state=optuna.trial.TrialState.PRUNED)
                    print(json.dumps({'trial': trial.number, 'state': 'pruned', 'epoch': row['epoch']}), flush=True)
                    del active[device]
            time.sleep(3)
    finally:
        for _, process, _, _ in active.values():
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait()
    completed = sorted((trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE),
                       key=lambda trial: trial.value, reverse=True)
    finalists = completed[:args.finalists]
    results = []
    for offset in range(0, len(finalists), len(args.devices)):
        jobs = []
        for device, trial in zip(args.devices, finalists[offset:offset + len(args.devices)]):
            name = trial.user_attrs['run_name'] + '_final'
            process = launch(name, trial.user_attrs['options'], device, args.final_epochs, out)
            jobs.append((process, name, trial.number))
        for process, name, number in jobs:
            if process.wait() != 0:
                raise RuntimeError(f'Finalist {name} failed')
            result = json.loads((ROOT / 'runs' / name / 'result.json').read_text())
            results.append({'name': name, 'trial': number, **result})
    summary = {'status': 'complete', 'trials': [
        {'number': trial.number, 'state': trial.state.name, 'value': trial.value,
         'params': trial.params, 'run_name': trial.user_attrs.get('run_name')}
        for trial in study.trials], 'finalists': results}
    (out / 'result.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
