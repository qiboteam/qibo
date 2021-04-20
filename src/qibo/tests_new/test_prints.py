"""Test that the source contains no prints."""
import os
import pytest
import pathlib


class CodeText:
    """Helper class to iterate through code text skipping the docstrings."""

    def __init__(self, code, filedir=None):
        self.code = code
        self.filedir = filedir
        self.line_counter = 0
        self.piece_counter = 0
        self.starts_with_docstring = (code[:3] == '"""')
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
            if self.piece_counter % 2 == self.starts_with_docstring:
                return self.pieces[self.piece_counter]
            else:
                return self.__next__()

        raise StopIteration

    def get_line(self, i): # pragma: no cover
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
            if i > 0: # pragma: no cover
                # This should not execute if the code does not contain `print` statements
                from qibo.config import raise_error
                line = self.get_line(i)
                raise_error(ValueError, f"Found `{target_word}` in line {line} "
                                        f"of {self.filedir}.")


def python_files():
    """Iterator that yields all python files (`.py`) in `/src/qibo/`."""
    basedir = pathlib.Path(os.path.realpath(__file__)).parent.parent
    for subdir, _, files in os.walk(basedir):
        for file in files:
            pieces = file.split(".")
            # skip non-`.py` files
            # skip current file because it contains `print`
            if len(pieces) == 2 and pieces[1] == "py" and pieces[0] != "test_prints":
                yield os.path.join(subdir, file)


@pytest.mark.parametrize("filename", python_files())
def test_qibo_code(filename):
    text = CodeText.from_file(filename)
    text.check_exists("print")
