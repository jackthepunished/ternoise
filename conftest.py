"""Guard against the editable-install finder silently serving another
checkout's code.

`python -m pytest` puts this checkout first on sys.path, but the venv's
editable install (a MetaPathFinder appended by setuptools, with absolute
paths into the main checkout) catches anything that falls through — e.g. a
module deleted or renamed on a worktree branch would silently import the main
checkout's stale copy and the suite would pass against the wrong code. For a
project whose review contract is bit-exactness, that is the worst failure
mode: green and wrong. So: every repo package the tests import must resolve
under this conftest's own directory, worktree or main checkout alike.
"""
from pathlib import Path


def pytest_sessionstart(session):
    import model
    import tools
    import vectors

    root = Path(__file__).resolve().parent
    for pkg in (model, tools, vectors):
        p = Path(pkg.__file__).resolve()
        assert root in p.parents, (
            f"{pkg.__name__} imported from {p}, outside the test root {root}: "
            "the editable-install finder is serving another checkout's code"
        )
