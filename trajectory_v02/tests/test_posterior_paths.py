"""Compare joint path draws with exhaustive enumeration, not marginal draws."""
import itertools
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from posterior_paths import sample_paths


class PathTests(unittest.TestCase):
    def test_enumerated_joint_distribution(self):
        transition = np.array([[.85,.1,.05],[.15,.75,.1],[.05,.2,.75]])
        emission = np.array([[.8,.2,.3],[.1,.8,.4],[.3,.5,.9]])
        initial = np.array([.7,.2,.1])
        combinations = list(itertools.product(range(3), repeat=3))
        joint = np.array([initial[a]*emission[0,a]*transition[a,b]*emission[1,b]
                          *transition[b,c]*emission[2,c] for a,b,c in combinations])
        joint /= joint.sum()
        filtered = [initial * emission[0]]
        for t in (1,2):
            filtered.append((filtered[-1] @ transition) * emission[t])
        uniforms = np.random.default_rng(20260921).random((20000,3))
        states = sample_paths(np.log(np.array(filtered)), np.log(transition), uniforms)
        observed = np.bincount(states @ np.array([9,3,1]), minlength=27) / len(states)
        # All 27 joint outcomes, including rare paths; deterministic fixed seed.
        tolerance = 6*np.sqrt(joint*(1-joint)/len(states)) + 2/len(states)
        self.assertTrue(np.all(np.abs(observed-joint) < tolerance))
        np.testing.assert_array_equal(states, sample_paths(np.log(np.array(filtered)),
                                                         np.log(transition), uniforms))

    def test_absorbing_chain_stays_constant(self):
        with np.errstate(divide='ignore'):
            transition = np.log(np.eye(3))
        filtered = np.tile(np.log([.2,.3,.5]), (8,1))
        states = sample_paths(filtered, transition, np.random.default_rng(5).random((100,8)))
        self.assertTrue(np.all(states == states[:,:1]))


if __name__ == '__main__':
    unittest.main()
