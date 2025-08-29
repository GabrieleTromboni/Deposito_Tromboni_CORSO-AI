import unittest
import importlib.util
import pathlib

# Load somma from the exact file path to avoid any module resolution issues
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
PROVE_PATH = PROJECT_ROOT / "prove.py"
spec = importlib.util.spec_from_file_location("prove_module", str(PROVE_PATH))
prove_module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(prove_module)
somma = prove_module.somma


class TestSomma(unittest.TestCase):
    def test_somma_positivi(self):
        self.assertEqual(somma(2, 3), 5)

    def test_somma_zero(self):
        self.assertEqual(somma(0, 0), 0)
        self.assertEqual(somma(0, 7), 7)
        self.assertEqual(somma(9, 0), 9)

    def test_somma_negativi(self):
        self.assertEqual(somma(-2, -3), -5)
        self.assertEqual(somma(-2, 3), 1)
        self.assertEqual(somma(2, -3), -1)

    def test_somma_grandi_numeri(self):
        self.assertEqual(somma(10**6, 10**6), 2 * 10**6)


if __name__ == "__main__":
    unittest.main()


