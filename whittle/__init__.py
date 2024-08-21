from pathlib import Path


def read_version():
    with open(Path(__file__).parent / "version") as f:
        return f.readline().strip().replace('"', "")


__version__ = read_version()
