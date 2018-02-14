
import sympy as sp 

import utils

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
    i20_transforms = utils.cumprod(transforms)
    Jw = sp.zeros(3, nq)
    for i, q in enumerate(qs):
        if q.name.startswith('t'):
            Jw[:,i] = i20_transforms[i][:-1,2]

    # insert into jacobian
    J = sp.zeros(6,6)
    J[:3,:nq] = Jv
    J[3:,:nq] = Jw
    return J
