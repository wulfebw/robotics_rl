
import collections
import sympy as sp 

import dh

def cumprod(ts):
    '''
    Description:
        - computes the cumulative product of a list of matrices

    Args:
        - ts: list of matrices, all square, all the same shape
    
    '''
    if len(ts) == 0:
        return []
    shape = ts[0].shape
    assert len(shape) == 2
    assert shape[0] == shape[1]
    side = shape[0]

    ret = []
    cur = sp.eye(side)
    for t in ts:
        cur = sp.simplify(cur * t)
        ret.append(cur)
    return ret

def jacobian(transforms, qs):
    '''
    Description:

    jacobian computation
    1. the linear velocity components of the jacobian, Jv 
        - to get these, differentiate the end effector point as defined in the  
            {0} frame (i.e., the last column of the transformation matrix)
        - with respect to each joint 
        - i.e., you have a (3x1) vector indicating the position (x,y,z) of the 
            end effector in the frame {0}
            + differentiate each of these x,y,z values wrt each joint 
                * so for each joint you end up with 3 terms 
        - the derivatives of the point wrt each joint, are the columns of a 
            (3x6) matrix Jv 

    2. the angular velocity components of the jacobian, Jw
        - if the joint is prismatic, then it is not associated with any angular 
            velocity change at it's column of the jacobian
        - so this is only for revolute joint 
        - for each revolute joint qi 
            + look at the transformation matrix (0<-i)T
            + get the z value from this transformation matrix 
            + and that's it

    Args:
        - transforms: list of transformation matrices from frame {N} to {0}
        - qs: variables in transforms to differentiate wrt
    '''
    nq = len(qs)
    
    # linear component of jacobian
    T = sp.simplify(sp.prod(transforms))
    p = T[:-1,-1] # end-effector position in frame {0}
    Jv = sp.zeros(3, nq)
    for i, q in enumerate(qs):
        Jv[:,i] = sp.diff(p, q, simplify=True)

    # angular component of jacobian
    i20_transforms = cumprod(transforms)
    Jw = sp.zeros(3, nq)
    for i, q in enumerate(qs):
        if q.name.startswith('t'):
            Jw[:,i] = i20_transforms[i][:-1,2]

    # insert into jacobian
    J = sp.zeros(6,6)
    J[:3,:nq] = Jv
    J[3:,:nq] = Jw
    return J
    
if __name__ == '__main__':
    # compute transformation matrix
    t1, t2, l1, l2 = sp.symbols('t1 t2 l1 l2')
    params = dict()
    params[1] = collections.defaultdict(int, dict(t=t1))
    params[2] = collections.defaultdict(int, dict(l=l1, t=t2))
    params[3] = collections.defaultdict(int, dict(l=l2))
    T, transforms = dh.build_transforms(params)
    sp.pprint(T)

    # compute jacobian
    qs = [t1, t2]
    J = jacobian(transforms, qs)
    sp.pprint(J)
    
