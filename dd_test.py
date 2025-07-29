import unittest
import torch as t
import numpy as np
from run_show_dd import Config, run

class TestSimple(unittest.TestCase):
    
    def test_basic_run_no_nan(self):
        """Basic test - make sure no NaN values"""
        cfg = Config(d=5, n=10, n_itr=10)
        results = run(cfg)
        
        # Check no NaN
        self.assertFalse(np.isnan(results['final_loss']))
        self.assertFalse(np.any(np.isnan(results['final_weights'])))
    
    def test_high_learning_rate(self):
        """Test high learning rate doesn't crash"""
        cfg = Config(d=3, n=5, n_itr=5, lr=1.0)
        results = run(cfg)
        
        # Should finish without error
        self.assertIsInstance(results['final_loss'], float)
    
    def test_zero_learning_rate(self):
        """Test zero learning rate"""
        cfg = Config(d=3, n=5, n_itr=5, lr=0.0, w_init=1.0)
        results = run(cfg)
        
        # Weights shouldn't change
        expected = np.zeros((3, 1))
        np.testing.assert_allclose(results['final_weights'], expected, atol=1e-6)
    
    def test_different_noise_types(self):
        """Test all noise types work"""
        # Input noise
        cfg1 = Config(d=3, n=5, n_itr=3, noise_type='input')
        results1 = run(cfg1)
        self.assertFalse(np.isnan(results1['final_loss']))
        
        # Output noise
        cfg2 = Config(d=3, n=5, n_itr=3, noise_type='output')
        results2 = run(cfg2)
        self.assertFalse(np.isnan(results2['final_loss']))
        
        # Time-correlated noise
        cfg3 = Config(d=3, n=5, n_itr=3, noise_type='time-correlated')
        results3 = run(cfg3)
        self.assertFalse(np.isnan(results3['final_loss']))

if __name__ == '__main__':
    unittest.main()
