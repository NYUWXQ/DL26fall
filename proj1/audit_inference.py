"""Check weight export and replay the CSV pipeline on labeled validation images."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image
import torch
from torchvision import transforms as T
from torchvision.datasets import CIFAR10

from predict import CompetitionImages
from train import ROOT, MEAN, STD, split_indices


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--holdout', type=Path, required=True)
    parser.add_argument('--full', type=Path, required=True)
    parser.add_argument('--export', type=Path, required=True)
    parser.add_argument('--submission', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    exported = torch.load(args.export, map_location='cpu', weights_only=False)
    full = torch.load(args.full, map_location='cpu', weights_only=False)
    holdout = torch.load(args.holdout, map_location='cpu', weights_only=False)
    assert full['config']['full_data'] and full['epoch'] == full['config']['epochs']
    assert full['config']['train_images'] == 50000 and not full['config']['max_batches']
    assert not holdout['config']['full_data']
    assert not exported['calibrate_bn']
    source = full['ema' if exported['weights'] == 'ema' else 'model']
    assert source.keys() == exported['model'].keys()
    assert all(torch.equal(value, exported['model'][name]) for name, value in source.items())
    data = CIFAR10(ROOT / 'data', train=True, transform=T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)]))
    assert data.classes == exported['classes']
    assert data.classes == ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    train, indices = split_indices(data.targets)
    split_hash = hashlib.sha256(train.tobytes() + indices.tobytes()).hexdigest()
    assert split_hash == holdout['config']['split_sha256']
    images = args.out / 'validation_png'
    (images / 'test_images').mkdir(parents=True)
    with (images / 'test.csv').open('w', newline='') as test, \
         (images / 'sample_submission.csv').open('w', newline='') as sample:
        test_writer, sample_writer = csv.writer(test), csv.writer(sample)
        test_writer.writerow(['id', 'filename'])
        sample_writer.writerow(['id', 'label'])
        for index in indices:
            name = f'{index}.png'
            Image.fromarray(data.data[index]).save(images / 'test_images' / name)
            test_writer.writerow([index, name])
            sample_writer.writerow([index, 0])
    decoded = CompetitionImages(images)
    assert all(torch.equal(decoded[i], data[int(index)][0]) for i, index in enumerate(indices))
    out = args.out / 'prediction' / 'submission.csv'
    subprocess.run([sys.executable, str(ROOT / 'predict.py'), str(args.holdout), '--data', str(images),
                    '--out', str(out), '--tta', exported['inference'],
                    '--precision', exported.get('inference_precision', 'bf16'), '--device', args.device], check=True)
    with out.open() as handle:
        rows = list(csv.DictReader(handle))
    assert [row['id'] for row in rows] == [str(index) for index in indices]
    labels = np.asarray(data.targets)[indices]
    accuracy = float(np.mean([int(row['label']) == label for row, label in zip(rows, labels)]))
    submitted = json.loads(args.submission.with_suffix('.json').read_text())
    assert hashlib.sha256(args.submission.read_bytes()).hexdigest() == submitted['submission_sha256']
    assert hashlib.sha256(args.export.read_bytes()).hexdigest() == submitted['checkpoint_sha256']
    with args.submission.open() as handle:
        actual_rows = list(csv.DictReader(handle))
    probabilities = np.load(args.submission.with_suffix('.probabilities.npy'))
    assert np.array_equal(np.asarray([int(row['label']) for row in actual_rows]), probabilities.argmax(1))
    report = {'status': 'passed', 'export_matches': exported['weights'], 'classes': data.classes,
              'validation_png_tensor_matches': len(indices), 'inference': exported['inference'],
              'precision': exported.get('inference_precision', 'bf16'),
              'csv_pipeline_validation_accuracy': accuracy,
              'submitted_csv_matches_saved_probabilities': True, 'submission_sha256': submitted['submission_sha256'],
              'full_training_epochs': full['epoch'], 'full_training_steps': full['steps'],
              'hidden_evaluation_images_read': False}
    (args.out / 'result.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
