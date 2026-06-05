from __future__ import annotations
from rich.logging import RichHandler
from rich.console import Console
import logging


def configure_logging() -> None:
    console = Console(stderr=True)
    
    handler = RichHandler(
        console=console,
        rich_tracebacks=True,
        tracebacks_show_locals=True,
        show_time=True,
        show_path=True,
        markup=True,
    )
    handler.setFormatter(logging.Formatter("%(message)s", datefmt="[%X]"))
    
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()
    root.addHandler(handler)
    
    for name in ("uvicorn", "uvicorn.access", "uvicorn.error"):
        logger = logging.getLogger(name)
        logger.handlers.clear()
        logger.propagate = True
