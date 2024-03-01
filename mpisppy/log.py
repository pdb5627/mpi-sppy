# Copyright 2020 by B. Knueven, D. Mildebrath, C. Muir, J-P Watson, and D.L. Woodruff
# Copyright 2024 by P. Brown (pdb5627)
# This software is distributed under the 3-clause BSD License.
"""
This is the logging configuration for mpisppy.

The documentation below is primarily for mpisppy developers.

Examples
========
To use the logger in your code, add the following 
after your import
.. code-block:: python
   
   import logging
   logger = logging.getLogger(__name__)

Then, you can use the standard logging functions
.. code-block:: python
   
   logger.debug('message')
   logger.info('message')
   logger.warning('message')
   logger.error('message')
   logger.critical('message')
   
Note that by default, any message that has a logging level
of warning or higher (warning, error, critical) will be
logged.

To log an exception and capture the stack trace
.. code-block:: python

   try:
      c = a / b
   except Exception as e:
      logging.error("Exception occurred", exc_info=True)


The current implementation of `global_toc` depends on variables referencing
loggers being named `logger`.

Logging configuration should be done by the application rather than by modules
intended to be imported from the package. Two utility functions are provided to
assist with configuring logging. An example configuration file is included with
the package and can be loaded using `load_default_config`.

A helper filter is provided to add elapsed time, as well as global rank, strata
rank, and cylinder rank to log records. See the example `logging_config.yml` file
for how to use it.

A helper handler is provided to add the global rank number to the file handler log
filename for per-rank logs. The tool toolong (https://github.com/Textualize/toolong)
can be used to merge timestamped logs or tail multiple log files simultaneously.

"""
import time
import inspect
from pathlib import Path
from typing import Optional, Union
import logging.config
import logging
import warnings
import yaml

from mpisppy.MPI import COMM_WORLD

default_timer = time.perf_counter
global_rank = COMM_WORLD.Get_rank()
global_tstart = default_timer()


def setup_logger(name, out, level=logging.DEBUG, mode='w', fmt=None):
   ''' Changed to do nothing
       TODO: Issue a deprecation warning
   '''
   warnings.warn("Logging should be configured at the application level. One possible way to configure logging is with `mpisppy.log.", DeprecationWarning)


def _reach(name):
    """ Find a variable by name in the call stack. Starts at the calling function works up the stack,
    looking for `name` or `self.name`."""
    for f in inspect.stack():
        if name in f[0].f_locals:
            return f[0].f_locals[name]
        if 'self' in f[0].f_locals and hasattr(f[0].f_locals['self'], name):
            return getattr(f[0].f_locals['self'], name)
    return None


def global_toc(msg, cond=(global_rank==0)):
    """ Legacy global_toc function for logging from rank0.
    It does its best to find a logger by looking for the name
    `logger` or `self.logger` in the call stack.
    """
    if cond:
        logger = _reach("logger")
        if not logger:
            logger = logging.getLogger("mpisppy")
        logger.info(msg)


class AddRanksAndElapsedTimeFilter():
    """
    Utility filter class to attach to handlers to add contextual information
    log records that can then be included in the log record formatting. Since
    filters do NOT propagate in Python logging, it is easiest to attach the
    to all handlers with formatters that may want to make use of the contextual
    information. See the `logging_config.yml` file for an example of usage.
    """
    def filter(self, record):
        # Add global_rank
        if not hasattr(record, "global_rank"):
            record.global_rank = global_rank

        # Add tictoc time
        if not hasattr(record, "elapsed_time"):
            record.elapsed_time = default_timer() - global_tstart

        # Try to find strata_rank and cylinder_rank from calling frames
        for attr in ["strata_rank", "cylinder_rank"]:
            if not hasattr(record, attr):
                attr_val = _reach(attr)
                if attr_val is not None:
                    setattr(record, attr, attr_val)

        # Format the ranks into a single string for including in the record format string
        if hasattr(record, "strata_rank") and hasattr(record, "cylinder_rank"):
            record.rank = f"R={record.global_rank},C={record.cylinder_rank},S={record.strata_rank}"
        else:
            record.rank = f"R={record.global_rank}"
        return True


class FileHandlerByRank(logging.FileHandler):
    def __init__(self, filename, *args, **kwargs):
        """ Returns a `FileHandler` object with `.{rank}` appended to the filename stem.
        Takes the same arguments as `FileHandler`.
        """
        # Convert filename to a Path object
        filename = Path(filename)
        rank_stem = filename.stem + '.' + str(global_rank)
        rank_filename = filename.with_stem(rank_stem)
        super().__init__(rank_filename, *args, **kwargs)


def load_default_config(logfile_dir: Optional[Union[Path, str]]=None):
    """ Load the default logging configuration that ships with the package.
    Args:
    logfile_dir: Directory to which relative FileHandler filenames will be relative. If not supplied,
        log filenames will be relative to the current working directory. Absolute filenames are not
        modified."""
    fpath = Path(__file__).parent / "logging_config.yml"
    load_config(fpath, logfile_dir)


def load_config(configfile: Union[Path, str], logfile_dir: Optional[Union[Path, str]]=None):
    """ Load a logging configuration from a YAML file
    Args:
    configfile: `str` or `Path` of logging YAML file from which to load logging configuration
    logfile_dir: Directory to which relative FileHandler filenames will be relative. If not supplied,
        log filenames will be relative to the current working directory. Absolute filenames are not
        modified."""
    # Read the text of the file
    config = Path(configfile).read_text()

    # Convert YAML document to a Python dict
    config = yaml.safe_load(config)

    if logfile_dir:
        # Update relative filenames
        handlers = config.get('handlers', dict())
        for name in handlers:
            filename = handlers[name].get('filename', None)
            if filename:
                filename = Path(filename)
                if not filename.is_absolute():
                    new_filename = logfile_dir / filename
                    handlers[name]['filename'] = new_filename

    # Configure logging using the loaded document
    logging.config.dictConfig(config)
