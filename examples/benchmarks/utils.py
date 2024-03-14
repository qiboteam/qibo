import json
import os


class BenchmarkLogger(list):
    def __init__(self, filename=None):
        self.filename = filename
        if filename is not None and os.path.isfile(filename):
            print(f"Extending existing logs from {filename}.")
            with open(filename) as file:
                super().__init__(json.load(file))
        else:
            if filename is not None:
                print(f"Creating new logs in {filename}.")
            super().__init__()

    def dump(self):
        if self.filename is not None:
            with open(self.filename, "w") as file:
                json.dump(list(self), file)

    def __str__(self):
        return "\n".join(f"{k}: {v}" for k, v in self[-1].items())


def parse_accelerators(accelerators):
    """Transforms string that specifies accelerators to dictionary.

    The string that is parsed has the following format:
        n1device1,n2device2,n3device3,...
    and is transformed to the dictionary:
        {'device1': n1, 'device2': n2, 'device3': n3, ...}

    Example:
        2/GPU:0,2/GPU:1 --> {'/GPU:0': 2, '/GPU:1': 2}
    """
    if accelerators is None:
        return None

    def read_digit(x):
        i = 0
        while x[i].isdigit():
            i += 1
        return x[i:], int(x[:i])

    acc_dict = {}
    for entry in accelerators.split(","):
        device, n = read_digit(entry)
        if device in acc_dict:
            acc_dict[device] += n
        else:
            acc_dict[device] = n
    return acc_dict
