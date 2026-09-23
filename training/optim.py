"""Optimiser construction with correct weight-decay handling.

Single definition, shared by every training loop in the project.

Why this module exists
----------------------
The coupled-L2 pattern ``torch.optim.Adam(model.parameters(), weight_decay=wd)``
appeared independently in at least six places: ``TemporalTrainer.__init__``,
four optimiser sites in ``notebooks/03_baseline_comparison.ipynb`` (including
``train_variant``, which produced every baseline checkpoint reported in Table I),
and ``code_release/training/trainer.py``. Fixing them one at a time guarantees
they drift apart again, which is exactly how two evaluation defects survived a
year in this project.

Two problems with that pattern
------------------------------
1. ``Adam`` adds the decay term to the gradient *before* adaptive scaling, so
   for a parameter whose task gradient is small the decay dominates the
   numerator and every step becomes a near-constant push toward zero.
   ``AdamW`` decouples it, applying decay directly to the weights instead.

2. It decays LayerNorm scales and biases, which should never be decayed.

Measured consequence, from ``notebooks/07_architecture_forensics.ipynb``
[Cell D1], on the ST-GNN + GRS checkpoints:

===========================  ==============  =========================
parameter group              at init         after training
===========================  ==============  =========================
LayerNorm gamma              exactly 1.0     rms 0.0018-0.0050, 58-62% denormal
GATv2 convolutions           std/init 1.0    std/init 0.0000, 98.9% denormal
proj (h_init, unnormalised)  std/init 1.0    std/init 0.30
===========================  ==============  =========================

The gamma collapse zeroes the message-passing branch output regardless of the
convolution weights; the convolutions, starved of gradient in consequence,
decay to denormal. [Cell D2] confirms the endpoint: zeroing every convolution
changes average precision by +0.0000 to +0.0007 with Spearman +1.0000, i.e. the
graph pathway is inert at inference.
"""

from typing import Iterable, List, Tuple

import torch


def split_decay_params(
    model: torch.nn.Module,
    extra_no_decay: Iterable[str] = ("residual_gates",),
) -> Tuple[List[torch.nn.Parameter], List[torch.nn.Parameter], List[str], List[str]]:
    """Partition parameters into decayed and decay-exempt groups.

    Exempt: every parameter of dimension <= 1 (all biases and all normalisation
    scales, which is the standard rule) plus any parameter whose qualified name
    contains one of ``extra_no_decay``.

    Returns ``(decay, no_decay, decay_names, no_decay_names)``; the names are
    returned so callers can log or assert on the split rather than trust it.
    """
    decay, no_decay, d_names, n_names = [], [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim <= 1 or any(k in name for k in extra_no_decay):
            no_decay.append(p)
            n_names.append(name)
        else:
            decay.append(p)
            d_names.append(name)
    return decay, no_decay, d_names, n_names


def build_optimizer(
    model: torch.nn.Module,
    lr: float,
    weight_decay: float,
    decoupled_wd: bool = False,
    verbose: bool = True,
) -> torch.optim.Optimizer:
    """Build the training optimiser.

    Parameters
    ----------
    decoupled_wd
        ``False`` (default) reproduces the original behaviour exactly: plain
        ``Adam`` with coupled L2 over ``model.parameters()``. Every published
        number depends on this remaining the default.
        ``True`` applies fix A1: ``AdamW`` with decoupled decay, exempting
        biases, normalisation scales and the residual gates.
    """
    if not decoupled_wd:
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    decay, no_decay, d_names, n_names = split_decay_params(model)
    if verbose:
        n_d = sum(p.numel() for p in decay)
        n_n = sum(p.numel() for p in no_decay)
        print(f'  [A1] AdamW decoupled wd={weight_decay}: '
              f'{len(decay)} tensors decayed ({n_d:,} params), '
              f'{len(no_decay)} exempt ({n_n:,} params)')
    leaked = [n for n in d_names
              if '.norms.' in n or 'init_norms' in n or 'residual_gates' in n]
    if leaked:
        raise RuntimeError(f'normalisation/gate parameters leaked into the decay group: {leaked}')
    return torch.optim.AdamW(
        [{'params': decay, 'weight_decay': weight_decay},
         {'params': no_decay, 'weight_decay': 0.0}],
        lr=lr,
    )
