import unittest
import tempfile
import json
import sys

class TestDashboardDataLoader(unittest.TestCase):
    def test_load_jsonl_empty(self):
        try:
            from dashboard.app import load_jsonl
            df = load_jsonl("non_existent_file.jsonl")
            self.assertTrue(df.empty)
        except ImportError:
            self.skipTest("Streamlit not installed")
        
    def test_load_jsonl_valid(self):
        try:
            from dashboard.app import load_jsonl
            with tempfile.NamedTemporaryFile("w", delete=False) as f:
                f.write(json.dumps({"a": 1}) + "\n")
                f.write(json.dumps({"a": 2}) + "\n")
                path = f.name
                
            df = load_jsonl(path)
            self.assertEqual(len(df), 2)
            self.assertEqual(df.iloc[0]["a"], 1)
        except ImportError:
            self.skipTest("Streamlit not installed")
