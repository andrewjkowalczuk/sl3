import numpy as np

def _ensure_unit_det(H: np.ndarray) -> np.ndarray:
    """Scales the elements of a homography matrix H such that det(H) = 1"""
    
    assert H.shape == (3, 3)

    # (Property) Determinant of an n x n matrix is a homogeneous function:
    # https://en.wikipedia.org/wiki/Determinant#Immediate_consequences
    # det(s * A) = s^n * det(A)
    # We want 1 = s^3 * det(H), hence s = det(A)^(-1/3)
    s = np.linalg.det(H) ** (-1.0 / 3)

    return s * H
