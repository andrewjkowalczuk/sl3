import numpy as np
from scipy.linalg import expm, logm

def _ensure_unit_det(H: np.ndarray) -> np.ndarray:
    """Scales the elements of a homography matrix H such that det(H) = 1"""
    
    assert H.shape == (3, 3)

    # (Property) Determinant of an n x n matrix is a homogeneous function:
    # https://en.wikipedia.org/wiki/Determinant#Immediate_consequences
    # det(s * A) = s^n * det(A)
    # We want 1 = s^3 * det(H), hence s = det(A)^(-1/3)
    s = np.linalg.det(H) ** (-1.0 / 3)

    return s * H

def exp(h: np.ndarray) -> np.ndarray:
    """Exponential map sl(3) -> SL(3)"""

    assert h.size == 8

    # Generators from "Homography-based Tracking for Central Catadioptric Cameras",
    # 2006, by Christoper Mei, Selim Benhimane, Ezio Malis, Patrick Rives:
    # G1 = [0, 0, 1; 0, 0, 0; 0, 0, 0]
    # G2 = [0, 0, 0; 0, 0, 1; 0, 0, 0]
    # G3 = [0, 1, 0; 0, 0, 0; 0, 0, 0]
    # G4 = [0, 0, 0; 1, 0, 0; 0, 0, 0]
    # G5 = [1, 0, 0; 0,-1, 0; 0, 0, 0]
    # G6 = [0, 0, 0; 0,-1, 0; 0, 0, 1]
    # G7 = [0, 0, 0; 0, 0, 0; 1, 0, 0]
    # G8 = [0, 0, 0; 0, 0, 0; 0, 1, 0]

    G = np.zeros((3, 3))
    G[0, 0] = h[4]
    G[0, 1] = h[2]
    G[0, 2] = h[0]
    G[1, 0] = h[3]
    G[1, 1] = -h[4] - h[5]
    G[1, 2] = h[1]
    G[2, 0] = h[6]
    G[2, 1] = h[7]
    G[2, 2] = h[5]

    return _ensure_unit_det(expm(G))

def log(H: np.ndarray) -> np.ndarray:
    """Logarithm SL(3) -> sl(3)"""

    G = logm(_ensure_unit_det(H))

    h = np.zeros(8)
    h[0] = G[0, 2]
    h[1] = G[1, 2]
    h[2] = G[0, 1]
    h[3] = G[1, 0]
    h[4] = G[0, 0]
    h[5] = G[2, 2]
    h[6] = G[2, 0]
    h[7] = G[2, 1]

    return h
