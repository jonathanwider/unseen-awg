import logging
import sys
import traceback
from functools import wraps
from typing import Any, Callable

from icecream import ic


def snakemake_handler(func: Callable) -> Callable:
    """
    Decorator to handle common Snakemake script patterns:
    - Redirect stdout/stderr to log files
    - Configure logging
    - Handle exceptions with proper logging
    - Ensure output streams are flushed
    """

    @wraps(func)
    def wrapper(snakemake: Any) -> None:
        # Redirect stdout and stderr if log files are specified
        if hasattr(snakemake, "log"):
            sys.stdout = open(snakemake.log.stdout, "w", buffering=1)
            sys.stderr = open(snakemake.log.stderr, "w", buffering=1)

        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            stream=sys.stderr,
        )

        try:
            # Call the actual function
            ic.configureOutput(outputFunction=lambda s: print(s, flush=True))
            func(snakemake)
        except Exception:
            logging.error("Script failed with error:")
            logging.error(traceback.format_exc())
            sys.exit(1)
        finally:
            sys.stdout.flush()
            sys.stderr.flush()

    return wrapper
