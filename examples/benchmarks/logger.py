import os
import json


class BenchmarkLogger(list):

    def __init__(self, filename=None):
        self.filename = filename
        if filename is not None and os.path.isfile(filename):
            print("Extending existing logs from {}.".format(filename))
            with open(filename, "r") as file:
                super().__init__(json.load(file))
        else:
            if filename is not None:
                print("Creating new logs in {}.".format(filename))
            super().__init__()

    def dump(self):
        if self.filename is not None:
            with open(self.filename, "w") as file:
                json.dump(list(self), file)

    def __str__(self):
        return "\n".join("{}: {}".format(k, v) for k, v in self[-1].items())
