
import collections
import numpy as np
import sympy as sp
import unittest

import rlrobo.dh

class TestDH(unittest.TestCase):

    def test_rr_manipulator(self):
        t1, t2, l1, l2 = sp.symbols('t1 t2 l1 l2')
        dh = dict()
        dh[1] = collections.defaultdict(int, dict(t=t1))
        dh[2] = collections.defaultdict(int, dict(l=l1, t=t2))
        dh[3] = collections.defaultdict(int, dict(l=l2))
        T, transforms = rlrobo.dh.build_transforms(dh)
        p = T[:-1,-1]
        self.assertEqual(p[0], l1 * sp.cos(t1) + l2 * sp.cos(t1 + t2))
        self.assertEqual(p[1], l1 * sp.sin(t1) + l2 * sp.sin(t1 + t2))
        self.assertEqual(T[:2,:2], sp.Matrix([[sp.cos(t1+t2),-sp.sin(t1 + t2)],[sp.sin(t1+t2), sp.cos(t1+t2)]]))
        
if __name__ == '__main__':
    unittest.main()