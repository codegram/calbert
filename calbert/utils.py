"Random utils used here and there"

__all__ = ["path_to_str"]

from pathlib import Path


def path_to_str(p: Path) -> str:
    "Converts a path into string, representing it as absolute if needed"
    return str(p.absolute())
