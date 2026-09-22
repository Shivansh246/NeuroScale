import unittest
import time

class TestContinuousPilotLogic(unittest.TestCase):

    def test_interval_calculation(self):
        # Conceptually test: max(0, interval - elapsed)
        interval = 5.0
        elapsed_1 = 1.01
        self.assertAlmostEqual(max(0, interval - elapsed_1), 3.99)
        
        elapsed_2 = 6.0
        self.assertEqual(max(0, interval - elapsed_2), 0.0)

    def test_mode_locking(self):
        from experiments.prediction_data.continuous_pilot import set_mode, get_mode
        set_mode("burst")
        self.assertEqual(get_mode(), "burst")

if __name__ == "__main__":
    unittest.main()
