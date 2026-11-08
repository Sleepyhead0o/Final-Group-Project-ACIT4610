import tempfile
import unittest

from pathlib import Path

import numpy as np

from problem2.portfolio import (
    load_data,
    repair,
    feasible,
    variance
)

from problem2.evolution_strategies import ALGS


class TestProblem2(unittest.TestCase):

    def setUp(self):
        """Create a small example portfolio"""

        self.r = np.array([
            0.02,
            0.06,
            0.10
        ])

        self.cov = np.diag([
            0.01,
            0.04,
            0.09
        ])

        self.min_r = 0.04

    def test_Parser(self):
        """Check dataset loading and covariance calculations"""

        with tempfile.TemporaryDirectory() as folder:

            path = Path(folder) / "small.txt"

            path.write_text(
                "3\n"
                "0.02 0.1\n"
                "0.06 0.2\n"
                "0.10 0.3\n"
                "1 1 1\n"
                "1 2 0\n"
                "1 3 0\n"
                "2 2 1\n"
                "2 3 0\n"
                "3 3 1\n"
            )

            r, cov = load_data(
                "small",
                folder
            )

            np.testing.assert_allclose(
                r,
                self.r
            )

            np.testing.assert_allclose(
                cov,
                self.cov
            )

    def test_Repair(self):
        """Check that invalid allocations are repaired"""

        w = repair(
            np.array([-0.4, 0.5, 0.9]),
            self.r,
            self.min_r
        )

        self.assertTrue(
            feasible(
                w,
                self.r,
                self.min_r
            )
        )

        self.assertAlmostEqual(
            float(w.sum()),
            1.0
        )

    def test_Variance(self):
        """Check portfolio variance"""

        w = np.array([
            0.5,
            0.3,
            0.2
        ])

        self.assertAlmostEqual(
            variance(w, self.cov),
            0.0097
        )

    def test_algorithms(self):
        """Check all three ES algorithms"""

        for name, alg in ALGS.items():

            with self.subTest(name=name):

                out = alg(
                    self.r,
                    self.cov,
                    self.min_r,
                    np.random.default_rng(42),
                    budget=95
                )

                # Check evaluation budget
                self.assertEqual(
                    out["evals"],
                    95
                )

                # Check feasibility
                self.assertTrue(
                    out["valid"]
                )

                # Best-so-far archive cannot worsen
                self.assertTrue(
                    np.all(
                        np.diff(out["hist"]) <= 1e-12
                    )
                )

                # Final archive value matches best result
                self.assertAlmostEqual(
                    out["hist"][-1],
                    out["var"]
                )


if __name__ == "__main__":
    unittest.main()