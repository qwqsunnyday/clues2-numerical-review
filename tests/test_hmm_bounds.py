"""Tiny real-kernel tests, run with NUMBA_BOUNDSCHECK=1 and fresh cache.

Without coal/ancient emissions, dense stochastic-matrix multiplication is an
independent reference for every backward row. Absorbing endpoint rows force the
upper-bound scan that the old implementation accessed out of range.
"""
import os
from pathlib import Path
import sys
import unittest
import numpy as np
from scipy.stats import norm
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))


class HMMBoundsTests(unittest.TestCase):
    def test_real_forward_backward_and_absorbing_states(self):
        if os.environ.get('NUMBA_BOUNDSCHECK')!='1':
            self.skipTest('Run the dedicated bounds gate with NUMBA_BOUNDSCHECK=1')
        import hmm_utils as hmm
        epochs=np.arange(6,dtype=float)
        ne=np.full(6,10000.)
        frequencies=np.linspace(0.,1.,15)
        logs=np.log(np.clip(frequencies,1e-12,1-1e-12))
        log1=np.log(np.clip(1-frequencies,1e-12,1-1e-12))
        bins=np.linspace(0,1,2000);bins[0]=1e-10;bins[-1]=1-1e-10
        z=norm.ppf(bins);cdf=norm.cdf(z)
        times=np.zeros((2,0));empty=np.zeros(0)
        ancient=np.zeros((0,4));haps=np.zeros((0,3))
        for s in (-.02,0.,.02):
            selection=np.full(6,s)
            trans=hmm._nstep_log_trans_prob(ne[0],s,frequencies,z,cdf,cdf,.5)
            prob=np.exp(trans)
            np.testing.assert_allclose(prob.sum(axis=1),1.,atol=1e-12)
            for current in (0.,.37,1.):
                b=hmm.backward_algorithm(selection,times,empty,empty,epochs,ne,.5,frequencies,
                    logs,log1,z,cdf,cdf,ancient,haps,trans,noCoals=1,precomputematrixboolean=1,currFreq=current)
                a=hmm.forward_algorithm(selection,times,empty,empty,epochs,ne,.5,frequencies,
                    logs,log1,z,cdf,cdf,ancient,haps,noCoals=1)
                expected=np.zeros(len(frequencies));expected[np.argmin(abs(frequencies-current))]=1
                for generation in range(5):
                    expected=expected@prob
                    np.testing.assert_allclose(np.exp(b[generation]),expected,atol=1e-12)
                # Uniform terminal factor remains uniform with a stochastic
                # transition; posterior evidence is constant at all five times.
                np.testing.assert_allclose(np.exp(a[1:]),1/len(frequencies),atol=1e-12)
                np.testing.assert_allclose(np.exp(a[1:]+b[:-1]).sum(axis=1),1/len(frequencies),atol=1e-12)


if __name__=='__main__':
    unittest.main()
