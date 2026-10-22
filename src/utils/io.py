import json
from pathlib import Path
from typing import Any


def get_root() -> Path:
    """
    Get the root directory of the project.

    Returns
    -------
    Path
        Root directory path.
    """
    ROOT = Path(__file__).absolute().resolve().parent.parent.parent
    return ROOT


def read_json(fname: str | Path) -> list[dict[str, Any]] | dict[str, Any]:
    """
    Read the given json file.

    Parameters
    ----------
    fname : str or Path
        Filename of the json file.

    Returns
    -------
    list[dict[str, Any]] or dict[str, Any]
        Loaded json.
    """
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=dict)


def write_json(content: Any, fname: str | Path) -> None:
    """
    Write the content to the given json file.

    Parameters
    ----------
    content : Any
        Content to write (must be JSON-friendly).
    fname : str or Path
        Filename of the json file.
    """
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)
