import unittest

import numpy as np

from Scripts.analyze_expert_shared_base import spectrum_errors, storage_bytes


class SharedBaseScreenTest(unittest.TestCase):
    def test_known_rank_error_and_factor_storage(self):
        matrix = np.diag([3., 4.])
        errors = spectrum_errors(matrix, np.array([4., 3.]), [0, 1, 2])
        self.assertEqual(errors, {"0": 1., "1": .6, "2": 0.})
        self.assertEqual(storage_bytes(64, 8, 2611200), 1227520)
        # The residual can exceed the target norm; rank-zero error may exceed 1.
        self.assertEqual(spectrum_errors(matrix, np.array([8., 6.]), [0])["0"], 2.)
