
import collections
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp 

import dh
import jacobian
import utils

def num_left_pseudoinv(m):
    m = np.array(m).astype(float)

def singular_jacobian(J, vars, vals, tol=1e-4):
    n_vars = len(vars)
    subs = dict(zip(vars, vals))
    J_num = np.array(J.evalf(subs=subs)[:,:n_vars]).astype(float)
    G = np.dot(J_num, J_num.T)
    if np.linalg.matrix_rank(G, tol=tol) < n_vars:
        return True 
    else:
        return False

def find_joint_config(
    J, 
    J_inv,
    qs,
    p_des, 
    q_cur, 
    T,
    constraints=None,
    max_iter=50,
    threshold=1e-2,
    init_step_size=1.,
    step_size_threshold=1e-8,
    epsilon=1e-2):
    '''
    Description:
        - Finds a joint configuration that achieves an end-effector position 
            of p_des if one exists assuming the current joint config is q_cur.
        - Parts of this function are specific to the 2-link robot and need to be 
            changed. For instance, the linear jacobian singularity check 
            and the reliance on the order of the qs.

    Args:
        - J: jacobian of manipulator
        - qs: symbolic variables for joints
        - p_des: desired end-effector position
        - q_cur: current joint configuration
        - T: transformation matrix, used to check solution and for p_cur
        - constraints: list of constraints on the joints
        - max_iter: maximum iterations of algorithm
        - threshold: l2 between p_des and p_cur at which to stop
        - init_step_size: initial step of each itr
        - step_size_threshold: value of step_size at which the algorithm gives up
        - epsilon: amount to perturb q_cur when a singularity is encountered
    '''
    # symbolic variables for left pseudoinverse of jacobian and end-effector position
    nq = len(qs)
    p = T[:-1,-1].col_join(sp.Matrix([[0],[0],[0]]))

    for itr in range(max_iter):

        # reduce step size until stepping the change of joint config direction 
        # yields an improvement
        step_size = init_step_size
        while step_size > step_size_threshold:
            # create subs dict with current joint config 
            q_subs = dict(zip(qs, q_cur))

            # compute inverse of jacobian
            J_cur_inv = J_inv.evalf(subs=q_subs)

            # compute delta q and the new end-effector position
            p_cur = p.evalf(subs=q_subs)
            dq = J_cur_inv * (p_des - p_cur)
            q_new = q_cur + step_size * dq
            new_q_subs = dict(zip(qs, q_new))
            p_new = p.evalf(subs=new_q_subs)

            # check if new position is better and keep if so
            if (p_new - p_des).norm() < (p_cur - p_des).norm():
                p_cur = p_new 
                q_cur = q_new 
                break 

            # otherwise, reduce the step size and check for singularities
            else:
                step_size = step_size / 2
                # only consider the linear jacobian for now
                if singular_jacobian(sp.Matrix(J[:3,:]), qs, q_cur):
                    rand = np.random.randn(*q_cur.shape) * epsilon
                    rand[nq:] = 0
                    q_cur += rand

        # if reached the destination, break
        if (p_cur - p_des).norm() < threshold:
            break

    # check that q_cur yields p_des within some threshold
    p_cur = T.evalf(subs=dict(zip(qs, q_cur)))[:-1,-1]
    if (p_cur - sp.Matrix(p_des[:3,:])).norm() < threshold:
        return np.array(q_cur).astype(float), True

    # otherwise, return the closest configuration we reached and a failure indicator
    else:
        return np.array(q_cur).astype(float), False

if __name__ == '__main__':
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
    T, transforms = dh.build_transforms(params)
    # insert constants
    T = T.subs(subs)
    sp.pprint(T)

    # compute jacobian
    qs = [t1, t2]
    J = jacobian.jacobian(transforms, qs)
    # insert constants
    J = J.subs(subs)
    sp.pprint(J)

    J_inv = sp.zeros(6,6)
    J_inv[:len(qs),:] = utils.left_pseudoinv(J[:,:len(qs)])

    # compute joint configuration required to achieve specific end-effector pos
    qs = [t1, t2]
    p_des = sp.Matrix([[-1, -1, 0, 0, 0, 0]]).T
    q_cur = sp.Matrix([[0, 0, 0, 0, 0, 0]]).T
    q_des, found = find_joint_config(J, J_inv, qs, p_des, q_cur, T)
    print(q_des)

    p_cur = T.evalf(subs=dict(zip(qs, q_des)))[:-1,-1]
    p_cur = np.array(p_cur).astype(float)
    p_des = np.array(p_des).astype(float)
    print(p_cur)
    plt.quiver(0, 0, p_des[0], p_des[1], scale=1)
    plt.quiver(0, 0, p_cur[0], p_cur[1], scale=1)
    plt.show()
