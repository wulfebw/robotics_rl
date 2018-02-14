
import collections
import numpy as np
import sympy as sp
import unittest

import rlrobo.dh
import rlrobo.jacobian

class TestJacobian(unittest.TestCase):

    def test_rr_manipulator(self):
        # compute transformation matrix
        t1, t2, l1, l2 = sp.symbols('t1 t2 l1 l2')
        params = dict()
        params[1] = collections.defaultdict(int, dict(t=t1))
        params[2] = collections.defaultdict(int, dict(l=l1, t=t2))
        params[3] = collections.defaultdict(int, dict(l=l2))
        T, transforms = rlrobo.dh.build_transforms(params)

        # compute jacobian
        qs = [t1, t2]
        J = rlrobo.jacobian.jacobian(transforms, qs)
        s = sp.sin
        c = sp.cos
        expected = sp.Matrix([
            [-l1 * s(t1) - l2 * s(t1+t2), -l2 * s(t1+t2)],
            [l1*c(t1)+l2*c(t1+t2),l2*c(t1+t2)]
        ])
        self.assertEqual(expected, J[:2,:2])
        self.assertEqual(1, J[-1,0])
        self.assertEqual(1, J[-1,1])
        
if __name__ == '__main__':
    unittest.main()