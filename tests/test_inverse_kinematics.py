
import collections
import numpy as np
import sympy as sp
import unittest

import rlrobo.dh
import rlrobo.inverse_kinematics
import rlrobo.jacobian
import rlrobo.utils

class TestJacobian(unittest.TestCase):

    def test_rr_manipulator(self):
        # symbols used throughout equations
        t1, t2, l1, l2 = sp.symbols('t1 t2 l1 l2')

        # manipulator constants
        subs = [
            (l1, 1),
            (l2, 1)
        ]

        # compute transformation matrix
        params = dict()
        params[1] = collections.defaultdict(int, dict(t=t1))
        params[2] = collections.defaultdict(int, dict(l=l1, t=t2))
        params[3] = collections.defaultdict(int, dict(l=l2))
        T, transforms = rlrobo.dh.build_transforms(params)
        # insert constants
        T = T.subs(subs)

        # compute jacobian
        qs = [t1, t2]
        J = rlrobo.jacobian.jacobian(transforms, qs)
        # insert constants
        J = J.subs(subs)

        J_inv = sp.zeros(6,6)
        J_inv[:len(qs),:] = rlrobo.utils.left_pseudoinv(J[:,:len(qs)])
        J_inv_fn = sp.lambdify(qs, J_inv)

        # compute joint configuration required to achieve specific end-effector pos
        p_des_list = [
            sp.Matrix([[-1, -1, 0, 0, 0, 0]]).T,
            sp.Matrix([[-1, 0, 0, 0, 0, 0]]).T,
            sp.Matrix([[-1, 1, 0, 0, 0, 0]]).T,
            sp.Matrix([[-2, 0, 0, 0, 0, 0]]).T,
            sp.Matrix([[0, -2, 0, 0, 0, 0]]).T,
            sp.Matrix([[1, 1, 0, 0, 0, 0]]).T,
            sp.Matrix([[1, -1, 0, 0, 0, 0]]).T,
        ]
        for p_des in p_des_list:
            qs = [t1, t2]
            q_cur = sp.Matrix([[0, 0, 0, 0, 0, 0]]).T
            q_des, found = rlrobo.inverse_kinematics.find_joint_config(
                J, J_inv_fn, qs, p_des, q_cur, T)
            p_cur = T.evalf(subs=dict(zip(qs, q_des)))[:-1,-1]
            p_cur = np.array(p_cur).astype(float)
            p_des = np.array(p_des).astype(float)
            np.testing.assert_array_almost_equal(p_des[:3], p_cur[:3], 1)
        
if __name__ == '__main__':
    unittest.main()