"""Optimiser construction with correct weight-decay handling.

Why this module exists
----------------------
The pattern ``torch.optim.Adam(model.parameters(), weight_decay=wd)`` is easy to write
and, in a gated architecture, can silently disable one branch of the model. Two things
go wrong with it.

1. ``Adam`` adds the decay term to the gradient *before* adaptive scaling, so for a
   parameter whose task gradient is small the decay dominates the numerator and every
   step becomes a near-constant push toward zero. ``AdamW`` decouples it, applying decay
   to the weights directly.

2. It decays LayerNorm scales and biases, which should not be decayed at all.

What that did to the checkpoints reported in the paper
------------------------------------------------------
Measured on the ST-GNN + GRS checkpoints:

===========================  ==============  =================================
parameter group              at init         after training
===========================  ==============  =================================
LayerNorm gamma              exactly 1.0     rms 0.0018-0.0050, 58-62% denormal
GATv2 convolutions           std/init 1.0    0.0% to 99.8% denormal by seed
proj (h_init, unnormalised)  std/init 1.0    std/init 0.30
===========================  ==============  =================================

The gamma collapse attenuates the message-passing branch output regardless of what the
convolutions hold, and the convolutions, starved of gradient in consequence, decay toward
denormal. The same decay reaches biases, so every sigmoid gate in the architecture ends at
the constant sigma(0) = 0.5.

WHAT THIS DOES AND DOES NOT TELL YOU. These are statements about the optimiser, not about
what a component contributes. Zeroing the convolutions in a checkpoint at 0.0% denormal
still moves average precision by +0.0404, so a branch can be heavily decayed and still
carry signal. Contribution is settled only by removing the component and re-scoring; the
paper reports that four of the model's five components fail that test.

``audit_checkpoint.py`` in the repository root reports these parameter statistics for a
saved model, needing neither data nor a GPU. Read it as a diagnosis of training dynamics,
and do not substitute it for an ablation.
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
