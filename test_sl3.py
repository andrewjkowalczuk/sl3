import unittest

import numpy as np

from sl3 import exp, log

class SL3Tests(unittest.TestCase):
    
    def test_exp_log(self):
        np.random.seed(54541)
        
        # exp() and log() are inverses of each other
        for i in range(100):
            H = exp(np.random.rand(8))
            self.assertTrue(np.allclose(H, exp(log(H))))

        # An all-zero vector in sl(3) corresponds to identity matrix in SL(3)
        self.assertTrue(np.allclose(np.eye(3), exp(np.zeros(8))))

if __name__ == "__main__":
    unittest.main()
