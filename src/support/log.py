import datetime as dt
import inspect
import logging as log
import os
import sys
from typing import Optional


class CustomFormatter(log.Formatter):
    # https://stackoverflow.com/questions/384076/how-can-i-color-python-logging-output

    def __init__(self, __format: str, colored: bool = True):
        super().__init__()

        if colored:
            grey = "\x1b[38;20m"
            yellow = "\x1b[33;20m"
            red = "\x1b[31;20m"
            bold_red = "\x1b[31;1m"
            reset = "\x1b[0m"
        else:
            grey = yellow = red = bold_red = reset = ""

        formats = {
            log.DEBUG: grey + __format + reset,
            log.INFO: grey + __format + reset,
            log.WARNING: yellow + __format + reset,
            log.ERROR: red + __format + reset,
            log.CRITICAL: bold_red + __format + reset,
        }

        self.formats = formats

    def format(self, record):
        log_fmt = self.formats.get(record.levelno)
        formatter = log.Formatter(log_fmt)
        return formatter.format(record)


def make_run_uid(application_type: Optional[str] = None) -> str:
    """
    Build the canonical run identifier used to name output folders.

    Format: ``<YYYYMMDD_HHMMSS.ffffff>__<application_type>_<hostname>``
    (the application type is omitted when None).

    Args:
        application_type (Optional[str], optional): An optional string to include in
            the identifier for better identification of the application type.

    Returns:
        str: The run identifier.
    """
    hostname = os.uname()[1]
    uid = dt.datetime.now().strftime("%Y%m%d_%H%M%S.%f_") + "_"

    if application_type is not None:
        uid += f"_{application_type}"

    uid += f"_{hostname}"
    return uid


def make_run_folder(
        base_folder: str = "out",
        application_type: Optional[str] = None,
        create: bool = True,
) -> str:
    """
    Create (unless disabled) and return the canonical run output folder,
    ``<base_folder>/<run_uid>/``, using the same naming scheme as
    :func:`initialize_log`.

    Args:
        base_folder (str, optional): The root output folder. Defaults to "out".
        application_type (Optional[str], optional): The application type tag
            embedded in the folder name. Defaults to None.
        create (bool, optional): If True, the folder is created on disk.
            Defaults to True.

    Returns:
        str: The path to the run folder, with a trailing slash.
    """
    folder = f"{base_folder.rstrip('/')}/{make_run_uid(application_type)}/"
    if create:
        os.makedirs(folder, exist_ok=True)
    return folder


def initialize_log(
        log_level: str = "DEBUG",
        name: str = "main",
        console_only: bool = False,
        application_type: Optional[str] = None,
        create_out_subfolders: bool = True,
) -> str | None:
    """
    This function initializes the logging system for the application. It sets up
    the log format, log level, and log handlers for both console and file
    output. The log files are organized in a structured manner based on the
    current date, time, hostname, and application type.

    Args:
        log_level (str, optional): The logging level to be set for the logger. Defaults to "DEBUG".
        name (str, optional): The prefix added in log lines and log file names. Defaults to "main".
        console_only (bool, optional): If True, logs will only be printed to the console
            and not saved to a file. Defaults to False.
        application_type (Optional[str], optional): An optional string to include in the log file name
         for better identification of the application type. Defaults to None.
        create_out_subfolders (bool, optional): If True, creates subfolders in the output directory for
         better organization of log files. Defaults to True.

    Returns:
        str: The path to the log folder where log files are saved, or None if console_only is True.
    """
    OUT_FOLDER: str = "out"

    uid = make_run_uid(application_type)

    if create_out_subfolders:
        OUT_FOLDER += f"/{uid}/"
        formatter = f"[{uid}] - %(asctime)s - %(levelname)s - %(message)s"
    else:
        OUT_FOLDER += "/"
        formatter = f"[audit] - %(asctime)s - %(levelname)s - %(message)s"

    os.makedirs(OUT_FOLDER, exist_ok=True)

    # Convert string to log level
    log_level: int | str
    try:
        log_level = getattr(log, log_level.upper())
    except AttributeError:
        log_level = log.DEBUG
        print(f"Invalid log level. Using default: {log_level}", file=sys.stderr)

    log.basicConfig(level=log_level, stream=sys.stdout, force=True)
    log.getLogger().handlers[0].setFormatter(CustomFormatter(formatter))
    log.getLogger().handlers[0].addFilter(
        lambda record: record.levelno < log.WARNING)
    # Send everything less than warning to stdout,
    # warnings and errors to stderr. Respect the chosen log level.

    stderr_handler = log.StreamHandler(sys.stderr)
    stderr_handler.setLevel(log_level)
    stderr_handler.setFormatter(CustomFormatter(formatter))
    stderr_handler.addFilter(lambda record: record.levelno >= log.WARNING)
    log.getLogger().addHandler(stderr_handler)

    if not console_only:
        # Add logging to log file
        lname = OUT_FOLDER + name + ".log"
        filehandler = log.FileHandler(lname, mode="a")
        filehandler.setLevel(log_level)
        filehandler.setFormatter(CustomFormatter(formatter, colored=False))
        log.getLogger().addHandler(filehandler)

    # Send initialization message to log
    log.info(f"Initialized logging with level {log_level} and output folder {OUT_FOLDER}")
    log.info(f"Log file name: {name}.log")
    log.info(f"Executable: {sys.argv[0]} {' '.join(sys.argv[1:])}")

    return OUT_FOLDER if not console_only else None


def silence_stdout_logging() -> None:
    global original_stdout_log_level
    original_stdout_log_level = log.getLogger().handlers[0].level
    log.getLogger().handlers[0].setLevel(log.CRITICAL)


def activate_stdout_logging() -> None:
    if "original_stdout_log_level" in globals():
        log.getLogger().handlers[0].setLevel(original_stdout_log_level)
    else:
        print("Logging was not silenced before", file=sys.stderr)


def tqdm_wrapper(iterable, **kwargs):
    from tqdm.auto import tqdm  # lazy import: keep the CLI below dependency-free

    # get name of calling function
    return tqdm(
        iterable,
        desc=f"Function {inspect.stack()[1][3]} cycling over a {type(iterable).__name__}",
        leave=True,
        file=sys.stdout,
        position=0,
        **kwargs,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Create a canonical OmniFlow run folder "
                    "(<base>/<YYYYMMDD_HHMMSS.ffffff>__<application_type>_<hostname>/) "
                    "and print its path. Used by shell entrypoints so that all "
                    "experiments share the same out-folder format.",
    )
    parser.add_argument(
        "--base", default="out",
        help="Base output folder (default: 'out').",
    )
    parser.add_argument(
        "--application-type", default=None,
        help="Application type tag embedded in the run folder name.",
    )
    cli_args = parser.parse_args()
    print(make_run_folder(cli_args.base, cli_args.application_type).rstrip("/"))
