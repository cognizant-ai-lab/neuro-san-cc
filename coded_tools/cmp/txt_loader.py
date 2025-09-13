# Copyright (C) 2023-2025 Cognizant Digital Business, Evolutionary AI.
# All Rights Reserved.
# Issued under the Academic Public License.
#
# You can be released from the terms, and requirements of the Academic Public
# License by purchasing a commercial license.
# Purchase of a commercial license is mandatory for any use of the
# neuro-san-studio SDK Software in commercial settings.
#
from typing import Any
from typing import Dict
from typing import Union
from pathlib import Path

from neuro_san.interfaces.coded_tool import CodedTool


class TxtLoader(CodedTool):
    """
    CodedTool implementation load a .txt file and return its content as string.
    """

    def invoke(self, args: Dict[str, Any], sly_data: Dict[str, Any]) -> Union[Dict[str, Any], str]:
        """
        :param args: An argument dictionary with the following keys
                file_path (str): The name of the .txt file to load.

        :param sly_data:
                None

        :return:
            If successful:
                The extracted text from the document.
            Otherwise:
                A text string error message in the format:
                "Error: <error message>"
        """
        print("############### Loading document ###############")
        file_path: str = args.get("file_path", "")
        print(f"File path: {file_path}")
        if not file_path:
            return "Error: No file path provided."

        # Build the file path
        full_file_path = Path.cwd() / Path(file_path)
        print(f"Full file path: {full_file_path}")

        # Extract text file content
        content = self.extract_txt_content(full_file_path)
        print("############### Document loading done ###############")
        return content

    @staticmethod
    def extract_txt_content(txt_path: Path) -> str:
        """
        Extract text from a plain text file.

        :param txt_path: Full path to the TXT file.
        :return: Content of the text file.
        """
        try:
            with open(txt_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            error = f"Error reading TXT {txt_path}: {e}"
            print(error)
            return error
