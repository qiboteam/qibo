"""Test that the source contains no prints."""

import os
import pathlib

import pytest

import qibo
from qibo.config import raise_error


class CodeText:
    """Helper class to iterate through code text skipping the docstrings."""

    def __init__(self, code, filedir=None):
        self.code = code
        self.filedir = filedir
        self.line_counter = 0
        self.piece_counter = 0
        self.pieces = None

    @classmethod
    def from_file(cls, filedir):
        assert filedir[-3:] == ".py"
        with open(filedir, encoding="utf-8") as file:
            code = file.read()
        return cls(code, filedir)

    def __iter__(self):
        self.line_counter = 0
        self.piece_counter = -1
        self.pieces = self.code.split('"""')
        return self

    def __next__(self):
        """Returns pieces of code text skipping the docstrings."""
        self.piece_counter += 1

        if self.piece_counter > 0:
            previous_piece = self.pieces[self.piece_counter - 1]
            self.line_counter += len(previous_piece.split("\n")) - 1

        if self.piece_counter < len(self.pieces):
            # skip docstrings
            if self.piece_counter % 2 == 0:
                return self.pieces[self.piece_counter]
            else:
                return self.__next__()

        raise StopIteration

    def check_exists(self, target_word):
        """Checks if a word exists in the code text.
        Raises a ValueError if the word is found.
        Ignores a specific occurence if `CodeText:skip` occurs on the same line.

        Args:
            target_word(str): word to be searched in the code text
        """
        for piece in self:
            for offset, line in enumerate(piece.split("\n")):
                if ("print" in line) and ("CodeText:skip" not in line):
                    line_num = self.line_counter + offset + 1
                    raise_error(
                        ValueError,
                        f"Found `{target_word}` in line {line_num} "
                        f"of {self.filedir}.",
                    )


def python_files():
    """Iterator that yields all python files (`.py`) in `/src/qibo/`."""
    basedir = pathlib.Path(qibo.__file__).parent.absolute()
    for subdir, _, files in os.walk(basedir):
        for file in files:
            pieces = file.split(".")
            # skip non-`.py` files
            # skip current file because it contains `print`
            if len(pieces) == 2 and pieces[1] == "py" and pieces[0] != "test_prints":
                yield os.path.join(subdir, file)


# Make sure the CodeText class works as intended
text_examples = [
    ('print("Test")', True),
    ('print("Test")\n""" docstring """', True),
    ('""" docstring """\nprint("Test")\n""" docstring """', True),
    ("pass", False),
    ('pass\n""" docstring with print """', False),
    ('""" docstring with print """\npass\n""" docstring with print """', False),
    ("pass # CodeText:skip", False),
    ('print("Test") # CodeText:skip', False),
    ('print("Test")\nprint("Test) # CodeText:skip', True),
    ('print("Test") # CodeText:skip\nprint("Test)', True),
]


@pytest.mark.parametrize(("text", "contains_print"), text_examples)
def test_codetext_class(text, contains_print):
    """Check if the CodeText class is working properly"""
    if contains_print:
        with pytest.raises(ValueError):
            CodeText(text).check_exists("print")
    else:
        CodeText(text).check_exists("print")


@pytest.mark.parametrize("filename", python_files())
def test_qibo_code(filename):
    text = CodeText.from_file(filename)
    text.check_exists("print")
