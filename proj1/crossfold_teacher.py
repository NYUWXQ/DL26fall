"""Five-fold probability teachers used only for full-data KL distillation."""
import hashlib
import json
import os
from pathlib import Path

import torch

from model import build_model


class CrossFoldTeacher(torch.nn.Module):
    def __init__(self, models, temperature):
        super().__init__()
        self.models = torch.nn.ModuleList(models)
        self.temperature = temperature

    def forward(self, images):
        target = torch.stack([(model(images).float() / self.temperature).softmax(1)
                              for model in self.models]).mean(0)
        return self.temperature * target.clamp_min(torch.finfo(target.dtype).tiny).log()


class ReplayTeacher(torch.nn.Module):
    """Replay frozen bf16 teachers; retain eager execution for unchecked inputs."""
    def __init__(self, model):
        super().__init__()
        assert not model.training and not any(p.requires_grad for p in model.parameters())
        self.model = model
        self.parameters_count = sum(p.numel() for p in model.parameters())
        self.captures = {}
        self.checked = set()
        self.eval()

    @staticmethod
    def signature(images):
        return tuple(images.shape), tuple(images.stride()), images.dtype, images.device

    def prepare(self, example):
        key = self.signature(example)
        if key in self.captures:
            return
        static = torch.empty_strided(example.shape, example.stride(),
                                     dtype=example.dtype, device=example.device)
        static.copy_(example)
        stream = torch.cuda.Stream(device=example.device)
        stream.wait_stream(torch.cuda.current_stream())
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            with torch.cuda.stream(stream):
                for _ in range(3):
                    self.model(static)
            stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                output = self.model(static)
        torch.cuda.current_stream().wait_stream(stream)
        self.captures[key] = (static, output, graph)

    def forward(self, images):
        key = self.signature(images)
        if (key not in self.captures or not torch.is_autocast_enabled('cuda')
                or torch.get_autocast_dtype('cuda') != torch.bfloat16):
            return self.model(images)
        assert not self.training and not self.model.training
        static, output, graph = self.captures[key]
        static.copy_(images)
        graph.replay()
        # KD may retain teacher tensors until backward, so never expose the replay buffer.
        result = output.clone()
        if key not in self.checked:
            with torch.no_grad():
                reference = self.model(images)
            equal = bool(torch.isfinite(result).all()) and torch.equal(result, reference)
            print(json.dumps({'teacher_graph_check': equal, 'job_id': os.environ.get('SLURM_JOB_ID'),
                              'group_parameters': self.parameters_count,
                              'shape': list(images.shape), 'stride': list(images.stride())}), flush=True)
            if not equal:
                self.captures.clear()
                return reference
            self.checked.add(key)
        return result


def recovery_graphs(model, device):
    # This execution-only change is approved for these two existing recovery jobs.
    if os.environ.get('SLURM_JOB_ID') not in ('21951012', '21963100'):
        return model
    record = {'teacher_graph_job': os.environ['SLURM_JOB_ID'],
              'group_parameters': sum(p.numel() for p in model.parameters()),
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    try:
        wrapped = ReplayTeacher(model)
        for batch in (128, 80):
            for layout in (torch.channels_last, torch.contiguous_format):
                example = torch.zeros((batch, 3, 32, 32), device=device).contiguous(memory_format=layout)
                wrapped.prepare(example)
        record.update(status='prepared', signatures=len(wrapped.captures))
    except Exception as error:
        wrapped = model
        record.update(status='eager_fallback', error=repr(error))
    print(json.dumps(record), flush=True)
    return wrapped


def load_crossfold_teacher(bundle, root, device, temperature):
    assert bundle['kind'] == 'cross_fold_teacher'
    members = bundle['members']
    assert len(members) == 5 and {member['fold'] for member in members} == set(range(5))
    models, hashes = [], []
    for member in members:
        path = root / 'runs' / member['path']
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == member['sha256']
        state = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
        config = state['config']
        assert config['training_data'] == 'CIFAR-10 official train split' and not config['max_batches']
        assert not config['full_data'] and config['train_images'] == 40000
        assert config['cv_fold'] == member['fold'] and config['split_sha256'] == member['split_sha256']
        assert state['epoch'] == config['epochs'] == member['epoch'] and state['weights'] == member['weights']
        assert not config.get('init_checkpoint') and not config.get('init_supervised')
        model = build_model(config['model']).to(device, memory_format=torch.channels_last)
        model.load_state_dict(state['model'])
        models.append(model.eval().requires_grad_(False))
        hashes.append(digest)
    assert sum(p.numel() for model in models for p in model.parameters()) == bundle['config']['parameters']
    teacher = CrossFoldTeacher(models, temperature).eval().requires_grad_(False)
    return recovery_graphs(teacher, device), hashes
