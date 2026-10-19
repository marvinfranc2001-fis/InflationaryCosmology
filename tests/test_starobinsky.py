import unittest
from modelos.starobinsky import sr1,sr2,primordial_scalar_power

class TestObservables(unittest.TestCase):
    def test_sr1(self):
        p=sr1(55)
        self.assertAlmostEqual(p.n_s,1-2/55-9/(2*55**2))
        self.assertAlmostEqual(p.r_005,12/55**2)
        self.assertEqual(p.alpha_s,0)
    def test_sr2(self):
        p=sr2(55)
        self.assertAlmostEqual(p.n_s,1-2/55-3/(2*55**2))
        self.assertAlmostEqual(p.r_005,12/55**2*(1-2/55))
        self.assertAlmostEqual(p.alpha_s,-2/55**2-3/55**3)
    def test_pivot(self):
        self.assertAlmostEqual(primordial_scalar_power(.05,2.1e-9,.965),2.1e-9)
    def test_bad_N(self):
        with self.assertRaises(ValueError): sr1(0)
        with self.assertRaises(ValueError): sr2(-4)
if __name__=="__main__":
    unittest.main()
