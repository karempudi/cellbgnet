import importlib.util
from pathlib import Path

try:
    import git
    _git_available = True
except ImportError:  # can cause import errors, not because of package but because of git
    _git_available = False

import cellbgnet


def cellbgnet_state() -> str:
    """Get version tag of decode. If in repo this will get you the output of git describe.
    Returns git describe, decode version or decode version with invalid appended.
    """

    p = Path(importlib.util.find_spec('cellbgnet').origin).parents[1]

    if _git_available:
        try:
            r = git.Repo(p)
            return "cellbgnet:" + cellbgnet.__version__  + ", current git commit at: " + r.head.object.hexsha

        except git.exc.InvalidGitRepositoryError:  # not a repo but an installed package
            return cellbgnet.__version__

        except git.exc.GitCommandError:
            return "vINVALID-recent-" + cellbgnet.__version__
    else:
        return "vINVALID-recent-" + cellbgnet.__version__


if __name__ == '__main__':
    v = cellbgnet_state()
    print(f"CELLBGNET version: {v}")