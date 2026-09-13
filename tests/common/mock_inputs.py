"""Deterministic synthetic logits for the local Mock workloads.

Each case samples uniform logits in [-1, 1], rounded to FP16 before the reference
calculation. Fixed seeds provide reproducible inputs with numerical sensitivity
and nonuniform matrices that require multiple normalization rounds.
"""
from __future__ import annotations

import numpy as np


PARAMETERS = {
    '1': {"shape": [1, 2, 4, 4], "seed": 8},
    '2': {"shape": [1, 4, 8, 8], "seed": 712},
    '3': {"shape": [1, 8, 8, 8], "seed": 2262},
    '4': {"shape": [2, 4, 8, 8], "seed": 1172},
    '5': {"shape": [1, 16, 6, 6], "matrix_seeds": [300456, 300293, 300991, 300131, 300353, 301666, 302436, 305262, 300716, 302894, 300418, 305033, 302362, 305408, 303168, 306659]},
}


def make_mock_logits(case: int, shape) -> np.ndarray:
    try:
        parameters = PARAMETERS[str(case)]
    except KeyError as error:
        raise ValueError(f"Unknown Mock input: {case}") from error
    if list(shape) != parameters["shape"]:
        raise ValueError("Mock input shape differs from its definition")
    if "matrix_seeds" in parameters:
        if len(parameters["matrix_seeds"]) != int(np.prod(shape[:-2])):
            raise ValueError("Mock matrix seed count differs from its shape")
        n = shape[-1]
        logits = np.stack([np.random.default_rng(seed).uniform(-1.0, 1.0, (n, n))
                           for seed in parameters["matrix_seeds"]]).reshape(shape)
    else:
        logits = np.random.default_rng(parameters["seed"]).uniform(-1.0, 1.0, size=shape)
    return np.ascontiguousarray(logits.astype(np.float16).astype(np.float32))
