import logging
import sys


def setup_logging(level: int = logging.INFO, prefix: str = "") -> None:
    root = logging.getLogger("pcgnn")
    root.setLevel(level)
    for h in list(root.handlers):
        if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler):
            root.removeHandler(h)
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter(f"%(asctime)s {prefix}%(levelname)s %(message)s", "%H:%M:%S"))
    root.addHandler(h)
    root.propagate = False
