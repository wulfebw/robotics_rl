
import collections
import copy
import sympy as sp

def R_z(t):
    s = sp.sin
    c = sp.cos
    r = sp.Matrix([
        [c(t),-s(t),0,0],
        [s(t),c(t),0,0],
        [0,0,1,0],
        [0,0,0,1]
    ])
    return r

def R_x(t):
    s = sp.sin
    c = sp.cos
    r = sp.Matrix([
        [1,0,0,0],
        [0,c(t),-s(t),0],
        [0,s(t),c(t),0],
        [0,0,0,1]
    ])
    return r

def D_z(l):
    r = sp.Matrix([
        [1,0,0,0],
        [0,1,0,0],
        [0,0,1,l],
        [0,0,0,1]
    ])
    return r

def D_x(l):
    r = sp.Matrix([
        [1,0,0,l],
        [0,1,0,0],
        [0,0,1,0],
        [0,0,0,1]
    ])
    return r

def build_transforms(dh):
    dh = copy.deepcopy(dh)
    links = list(sorted(dh.keys()))
    transforms = []
    for l in links:
        t = R_x(dh[l]['a']) * D_x(dh[l]['l']) * R_z(dh[l]['t']) * D_z(dh[l]['d'])
        transforms.append(t)
    return sp.simplify(sp.prod(transforms)), transforms

if __name__ == '__main__':
    t1, t2, l1, l2 = sp.symbols('t1 t2 l1 l2')
    dh = dict()
    dh[1] = collections.defaultdict(int, dict(t=t1))
    dh[2] = collections.defaultdict(int, dict(l=l1, t=t2))
    dh[3] = collections.defaultdict(int, dict(l=l2))
    T, transforms = build_transforms(dh)
    [sp.pprint(t) for t in transforms]
    sp.pprint(T)
    subs = [
        (t1, sp.rad(45)),
        (t2, sp.rad(90)),
        (l1, 1),
        (l2, 1)
    ]
    sp.pprint(T.subs(subs))