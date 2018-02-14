
import sympy as sp
import unittest

import rlrobo.utils as utils

class TestUtils(unittest.TestCase):

    def test_cumprod(self):
        x = sp.Matrix([
            [1,2],
            [3,4]
        ])
        prods = utils.cumprod([x,x,x])
        self.assertEquals(prods[0], sp.Matrix([[1,2],[3,4]]))
        self.assertEquals(prods[1], sp.Matrix([[7,10],[15,22]]))
        self.assertEquals(prods[2], sp.Matrix([[37,54],[81,118]]))

if __name__ == '__main__':
    unittest.main()