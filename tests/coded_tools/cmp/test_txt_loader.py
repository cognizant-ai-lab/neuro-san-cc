# Copyright (C) 2023-2025 Cognizant Digital Business, Evolutionary AI.
# All Rights Reserved.
# Issued under the Academic Public License.
#
# You can be released from the terms, and requirements of the Academic Public
# License by purchasing a commercial license.
# Purchase of a commercial license is mandatory for any use of the
# neuro-san-studio SDK Software in commercial settings.
#
from unittest import TestCase

from coded_tools.cmp.txt_loader import TxtLoader


class TestTxtLoader(TestCase):
    """
    Tests the TxtLoader CodedTool and makes sure it can load a .txt file and return its content as string.
    """

    def test_invoke(self):
        """
        Tests the invoke method of the OrderAPI CodedTool.
        Checks the response is correctly generated when all params are provided and valid.
        """
        loader = TxtLoader()
        args = {"file_path": "documents/CMP/CMP_txt/CMP2014_10 Decisions_1_to_8.txt"}
        content = loader.invoke(args=args, sly_data={})
        # Check the first line
        first_line = content.splitlines()[0]
        expected_first_line = "|  | United Nations | FCCC/KP/CMP/2014/9/Add.1 |"
        self.assertEqual(expected_first_line, first_line)
