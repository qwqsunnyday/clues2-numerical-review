"""Analytical regression cases for bugs that produced contradictory s and CI."""
import math
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from profile_search import fit_profile


class ProfileTests(unittest.TestCase):
    grid = [i/1000 for i in range(-100,101,10)]

    def test_negative_peak_and_known_interval(self):
        mean,sd=-.035,.009
        f=lambda x:-.5*((x-mean)/sd)**2+.5*(mean/sd)**2
        r=fit_profile(f,self.grid,xtol=1e-8)
        self.assertAlmostEqual(r['best_s'],mean,places=6)
        c=r['confidence_components'][0]
        self.assertAlmostEqual(c['lower'],mean-1.95996398454*sd,places=6)
        self.assertAlmostEqual(c['upper'],mean+1.95996398454*sd,places=6)

    def test_two_modes_keep_two_intervals(self):
        def f(x):
            return max(5-.5*((x+.05)/.004)**2,6-.5*((x-.04)/.004)**2)
        r=fit_profile(f,self.grid,xtol=1e-8)
        self.assertAlmostEqual(r['best_s'],.04,places=6)
        self.assertEqual(len(r['confidence_components']),2)

    def test_boundary_is_retained_and_censored(self):
        seen=[]
        def f(x):
            self.assertLessEqual(abs(x),.1)
            seen.append(x)
            return 100*x
        r=fit_profile(f,self.grid)
        self.assertEqual(r['best_s'],.1)
        self.assertTrue(r['optimum_at_boundary'])
        self.assertTrue(r['confidence_components'][0]['upper_censored'])
        self.assertIn(.1,seen)

    def test_flat_is_not_given_a_narrow_gaussian(self):
        r=fit_profile(lambda x:0.,self.grid)
        self.assertTrue(r['flat_over_initial_grid'])
        self.assertEqual(r['best_s'],0.)
        self.assertEqual(r['confidence_components'],[dict(lower=-.1,upper=.1,lower_censored=True,upper_censored=True)])

    def test_no_report_lower_than_observed_peak(self):
        f=lambda x:max(7-.5*((x+.07)/.01)**2,2-.5*((x-.03)/.01)**2)
        r=fit_profile(f,self.grid)
        self.assertGreaterEqual(r['logLR'],max(f(x) for x in self.grid))
        self.assertTrue(r['confidence_contains_best'])

    def test_nonfinite_is_rejected(self):
        with self.assertRaises(ValueError):
            fit_profile(lambda x:math.nan,self.grid)


if __name__=='__main__':
    unittest.main()
