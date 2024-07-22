import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from sl3 import exp, log

def main():
    rng = np.random.default_rng(54547)
    num_points = 100
    source_points = np.ones((3, num_points))
    source_points[0:2, :] = rng.standard_normal((2, num_points))

    # A homography matrix can be decomposed into a product of an upper triangular 
    # and an orthogonal matrix. This fact comes handy when constructing a test transform, 
    # allowing direct control over rotation parameters.
    homography = np.array([
        [1.1, 0.0, 0.1],
        [0.0, 0.9, 0.2],
        [0.0, 0.0, 1.0]]
        ) @ Rotation.from_euler('XYZ', [0.025, 0.01, 0.25]).as_matrix()

    target_points = homography @ source_points
    target_points /= target_points[2, :]

    def cost_func(h):
        pts = exp(h) @ source_points
        pts /= pts[2, :]
        return (target_points[0:2, :] - pts[0:2, :]).flatten()
    
    # Initial guess is a zero 8-vector, i.e., identity element in sl(3)
    result = least_squares(cost_func, np.zeros((8)))

    solution = exp(result.x)
    
    print(result)

    def calc_errors(transform):
        test_points = transform @ source_points 
        test_points /= test_points[2, :]
        return np.sqrt(np.sum(np.square(test_points[0:2, :] - target_points[0:2, :]), axis=0))

    diff = log(np.linalg.inv(homography) @ solution)
    
    print('Initial error = {}, final error = {}, sl(3) diff norm = {}'.format(
        np.mean(calc_errors(np.eye(3))),
        np.mean(calc_errors(solution)),
        np.linalg.norm(diff))
    )

    pts = solution @ source_points
    pts /= pts[2, :]

    _, ax = plt.subplots(1, 2, figsize=(10, 5))

    ax[0].plot(source_points[0, :], source_points[1, :], 'r.')
    ax[0].plot(target_points[0, :], target_points[1, :], 'go')
    ax[0].plot(
        np.vstack((target_points[0, :], source_points[0, :])), 
        np.vstack((target_points[1, :], source_points[1, :])), 
        'r-')
    ax[0].set_xlim([-2.75, 2.75])
    ax[0].set_ylim([-2.75, 2.75])
    ax[0].invert_yaxis()
    ax[0].set_aspect('equal')
    ax[0].set_title('Initial guess')
    
    ax[1].plot(pts[0, :], pts[1, :], 'go')
    ax[1].plot(target_points[0, :], target_points[1, :], 'r.')
    ax[1].set_xlim([-2.75, 2.75])
    ax[1].set_ylim([-2.75, 2.75])
    ax[1].invert_yaxis()
    ax[1].set_aspect('equal')
    ax[1].set_title('Final solution')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
