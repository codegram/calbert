"Random utils used here and there"

__all__ = ["normalize_path"]

from pathlib import Path
from hydra.utils import to_absolute_path


def normalize_path(p: Path) -> Path:
    "Converts a path into absolute gathering Hydra's original directory"
    try:
        return Path(to_absolute_path(str(p)))
    except AttributeError:  # if we're not in Hydra
        return p.absolute()
