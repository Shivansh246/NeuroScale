import unittest
from unittest.mock import patch, MagicMock
from experiments.prediction_data.long_run import get_mode, set_mode

class TestLongRun(unittest.TestCase):
    def test_mode_switch(self):
        set_mode("idle")
        self.assertEqual(get_mode(), "idle")
        set_mode("cpu")
        self.assertEqual(get_mode(), "cpu")

if __name__ == '__main__':
    unittest.main()
