from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def complex_statevector_to_ri(psi: ArrayLike) -> NDArray[np.float64]:
    psi_c = np.asarray(psi, dtype=np.complex128)
    if psi_c.ndim != 1:
        raise ValueError("psi must be a 1D complex array of length 2^n")
    out = np.empty((psi_c.shape[0], 2), dtype=np.float64)
    out[:, 0] = psi_c.real
    out[:, 1] = psi_c.imag
    return out


def ri_to_complex_statevector(psi_ri: ArrayLike) -> NDArray[np.complex128]:
    arr = np.asarray(psi_ri, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError("psi_ri must have shape (2^n, 2)")
    return arr[:, 0].astype(np.complex128) + 1j * arr[:, 1]


def complex_matrix_to_ri(mat: ArrayLike) -> NDArray[np.float64]:
    mat_c = np.asarray(mat, dtype=np.complex128)
    if mat_c.ndim != 2:
        raise ValueError("mat must be a 2D complex array")
    out = np.empty((mat_c.shape[0], mat_c.shape[1], 2), dtype=np.float64)
    out[:, :, 0] = mat_c.real
    out[:, :, 1] = mat_c.imag
    return out


def ri_to_complex_matrix(mat_ri: ArrayLike) -> NDArray[np.complex128]:
    arr = np.asarray(mat_ri, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[2] != 2:
        raise ValueError("mat_ri must have shape (N, M, 2)")
    return arr[:, :, 0].astype(np.complex128) + 1j * arr[:, :, 1]


def kraus_1q_to_ri(kraus_ops: ArrayLike) -> NDArray[np.float64]:
    """
    Convert a set of 1-qubit Kraus operators into the Rust-kernel format.

    Input shape: (k, 2, 2) complex
    Output shape: (k, 2, 2, 2) float64 with last dim [re, im]
    """
    k = np.asarray(kraus_ops, dtype=np.complex128)
    if k.ndim != 3 or k.shape[1:] != (2, 2):
        raise ValueError("kraus_ops must have shape (k, 2, 2) complex")
    out = np.empty((k.shape[0], 2, 2, 2), dtype=np.float64)
    out[:, :, :, 0] = k.real
    out[:, :, :, 1] = k.imag
    return out


