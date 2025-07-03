from functools import partial
from typing import Callable

from adv_lib.attacks.fast_minimum_norm import fmn
from adv_lib.attacks.stochastic_sparse_attacks import vfga
from adv_lib.attacks.primal_dual_gradient_descent import pdpgd
from utils.attack_wrappers import sparsefool, PGD0, sparse_rs, dataset_BB_attack, \
    BB_attack, EAD_attack, sparse_PGD
from sigma_zero_attack import sigma_zero

test_attacks = {
    'DTBB': partial(dataset_BB_attack),
    'BB': partial(BB_attack),
    'fixed-PGD0': partial(PGD0),
    'fixed-Sparse-RS': partial(sparse_rs),
    'sparse-PGD': partial(sparse_PGD),
    'SPARSEFOOL': partial(sparsefool),
    'EAD': partial(EAD_attack),
    'VFGA': partial(vfga),
    'PDPGD': partial(pdpgd),
    'FMN': partial(fmn),
    'sigma_zero': partial(sigma_zero),
    'sigma_zero*': partial(sigma_zero, sigma=1.0, threshold=0.1, lr=0.1, t=0.0015),
}


def get_attack(name: str) -> Callable:
    if name in test_attacks:
        return test_attacks[name]
    else:
        raise ValueError("Unknown attack: " + name)