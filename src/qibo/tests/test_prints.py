"""Test that the source contains no prints."""
import os
import pytest
import pathlib
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
        with open(filedir, "r") as file:
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

    def get_line(self, i):
        """Calculates line number of the identified `print`.

        Args:
            i (int): String index that the `print` was found.

        Returns:
            The line number of the corresponding `print` in the file.
        """
        piece = self.pieces[self.piece_counter][:i]
        return self.line_counter + len(piece.split("\n"))

    def check_exists(self, target_word):
        """Checks if a word exists in the code text."""
        for piece in self:
            i = piece.find(target_word)
            if i >= 0:
                # This should not execute if the code does not contain `print` statements
                line = self.get_line(i)
                raise_error(ValueError, f"Found `{target_word}` in line {line} "
                                        f"of {self.filedir}.")


def python_files():
    """Iterator that yields all python files (`.py`) in `/src/qibo/`."""
    excluded_files = ["tests/test_prints.py", "parallel.py"] # list with excluded files
    basedir = pathlib.Path(os.path.realpath(__file__)).parent.parent
    for subdir, _, files in os.walk(basedir):
        for file in files:
            pieces = file.split(".")
            full_path = os.path.join(subdir, file) # get the absolute path
            relative_path = full_path.split("/src/qibo/")[-1] # relative path from /src/qibo/
            # skip non-`.py` files i.e. pieces should be ["filename", ".py"]
            # skip excluded files i.e. relative path should not be in `excluded_files`
            if len(pieces) == 2 and pieces[1] == "py" and (relative_path not in excluded_files):
                yield full_path

# Make sure the CodeText class works as intended
text_examples = [
    ('print("Test")', True),
    ('print("Test")\n""" docstring """', True),
    ('""" docstring """\nprint("Test")\n""" docstring """', True),
    ('pass', False),
    ('pass\n""" docstring with print """', False),
    ('""" docstring with print """\npass\n""" docstring with print """', False)
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
