"""RLD and EA-KD losses; author revisions and adaptation details in modern_sources.json.

The fixed first experiments use unmixed labels and one locally trained teacher.
"""
import torch
from torch.nn import functional as F


def rld_loss(student, teacher, labels, temperature=4.):
    student, teacher = student.float(), teacher.detach().float()
    # SCD: transfer teacher confidence to the true student class, even on teacher errors.
    student_mask = torch.zeros_like(student, dtype=torch.bool).scatter_(1, labels[:, None], True)
    teacher_mask = torch.zeros_like(teacher, dtype=torch.bool).scatter_(1, teacher.argmax(1)[:, None], True)
    log_ps, pt = student.log_softmax(1), teacher.softmax(1)
    log_binary = torch.cat([log_ps.gather(1, labels[:, None]),
                            log_ps.masked_fill(student_mask, -torch.inf).logsumexp(1, keepdim=True)], 1)
    binary_teacher = torch.stack([(pt * teacher_mask).sum(1), (pt * ~teacher_mask).sum(1)], 1)
    scd = F.kl_div(log_binary, binary_teacher, reduction='batchmean')
    # MCD: exclude classes ranked at or above the true label by the teacher.
    mask = teacher >= teacher.gather(1, labels[:, None])
    ls = (student / temperature).masked_fill(mask, -1e9).log_softmax(1)
    pt = (teacher / temperature).masked_fill(mask, -1e9).softmax(1)
    mcd = F.kl_div(ls, pt, reduction='batchmean') * temperature**2
    return scd + 8 * mcd


def ea_loss(student, teacher, temperature=4.):
    student, teacher = student.float(), teacher.detach().float()
    kl = F.kl_div((student / temperature).log_softmax(1),
                  (teacher / temperature).softmax(1), reduction='none').sum(1) * temperature**2
    with torch.no_grad():
        ps, pt = (student / 3.).softmax(1), (teacher / 3.).softmax(1)
        hs = -(ps * (ps + 1e-10).log()).sum(1)
        ht = -(pt * (pt + 1e-10).log()).sum(1)
        # Follow the author's CIFAR code: H_ub is class count (100 -> 10 here).
        weight = (ht + ht * hs / student.shape[1]) / 2
    return (weight * kl).mean()
