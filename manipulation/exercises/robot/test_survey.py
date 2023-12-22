import hashlib
import unittest

import timeout_decorator
from gradescope_utils.autograder_utils.decorators import weight


class TestSurvey(unittest.TestCase):
    def __init__(self, test_name, notebook_locals):
        super().__init__(test_name)
        self.notebook_locals = notebook_locals

    @weight(1)
    @timeout_decorator.timeout(1.0)
    def test_dynamics(self):
        """Test Survey Code"""
        survey_code = self.notebook_locals["survey_code"]
        m = hashlib.sha1(survey_code.encode("utf-8"))
        self.assertEqual(
            m.hexdigest(),
            "415133a1dd0d559a2fbe766054892d0a6b16fee9",
            "wrong survey code!",
        )
