"""Logging and Profiling."""
from datetime import datetime
from time import time as get_time

from packaging.version import parse

from anndata.logging import get_memory_usage

from scMethtools import settings

_VERBOSITY_LEVELS_FROM_STRINGS = {"error": 0, "warn": 1, "info": 2, "hint": 3}


# TODO: Add docstrings
def info(*args, **kwargs):
    """TODO."""
    kwargs['time'] = kwargs.get('time', True)
    return msg(*args, v="info", **kwargs)


# TODO: Add docstrings
def error(*args, **kwargs):
    """TODO."""
    args = ("Error:",) + args
    return msg(*args, v="error", **kwargs)


# TODO: Add docstrings
def warn(*args, **kwargs):
    """TODO."""
    args = ("WARNING:",) + args
    return msg(*args, v="warn", **kwargs)


# TODO: Add docstrings
def hint(*args, **kwargs):
    """TODO."""
    return msg(*args, v="hint", **kwargs)


# TODO: Add docstrings
def _settings_verbosity_greater_or_equal_than(v):
    """TODO."""
    if isinstance(settings.verbosity, str):
        settings_v = _VERBOSITY_LEVELS_FROM_STRINGS[settings.verbosity]
    else:
        settings_v = settings.verbosity
    return settings_v >= v


def msg(
    *msg,
    v=None,
    time=False,
    memory=False,
    reset=False,
    end="\n",
    no_indent=False,
    t=None,
    m=None,
    r=None,
):
    r"""Write message to logging output.

    Log output defaults to standard output but can be set to a file
    by setting `scm.settings.log_file = 'mylogfile.txt'`.

    Parameters
    ----------
    v : {'error', 'warn', 'info', 'hint'} or int, (default: 4)
        0/'error', 1/'warn', 2/'info', 3/'hint', 4, 5, 6...
    time, t : bool, optional (default: False)
        Print timing information; restart the clock.
    memory, m : bool, optional (default: False)
        Print memory information.
    reset, r : bool, optional (default: False)
        Reset timing and memory measurement. Is automatically reset
        when passing one of ``time`` or ``memory``.
    end : str (default: '\n')
        Same meaning as in builtin ``print()`` function.
    no_indent : bool (default: False)
        Do not indent for ``v >= 4``.
    """
    # variable shortcuts
    if t is not None:
        time = t
    if m is not None:
        memory = m
    if r is not None:
        reset = r
    if v is None:
        v = 4
    if isinstance(v, str):
        v = _VERBOSITY_LEVELS_FROM_STRINGS[v]
    if v == 3:  # insert "--> " before hints
        msg = ("-->",) + msg
    if v >= 4 and not no_indent:
        msg = ("   ",) + msg
    if _settings_verbosity_greater_or_equal_than(v):
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        msg = (f"[{current_time}]","\t") + msg  # Add current time to the message
        if not time and not memory and len(msg) > 0:
            _write_log(*msg, end=end)
        if reset:
            try:
                settings._previous_memory_usage, _ = get_memory_usage()
            except ImportError as e:
                ImportError(e)
            settings._previous_time = get_time()
        if time:
            elapsed = get_passed_time()
            msg = msg + (f"({_sec_to_str(elapsed)})",)
            _write_log(*msg, end=end)
        if memory:
            _write_log(get_memory_usage(), end=end)


m = msg


# TODO: Add docstrings
def _write_log(*msg, end="\n"):
    """Write message to log output, ignoring the verbosity level.

    This is the most basic function.

    Parameters
    ----------
    msg
        One or more arguments to be formatted as string. Same behavior as print function.
    """
    from .settings import logfile
    
    from .settings import log_dir

    if logfile == "":
        print(*msg, end=end)
    else:
        out = ""
        for s in msg:
            out += f"{s} "
        with open(logfile, "a") as f:
            f.write(out + end)


# TODO: Add docstrings
def _sec_to_str(t, show_microseconds=False):
    """Format time in seconds.

    Parameters
    ----------
    t : int
        Time in seconds.
    """
    from functools import reduce

    t_str = "%d:%02d:%02d.%02d" % reduce(
        lambda ll, b: divmod(ll[0], b) + ll[1:], [(t * 100,), 100, 60, 60]
    )
    return t_str if show_microseconds else t_str[:-3]


# TODO: Add docstrings
def get_passed_time():
    """TODO."""
    now = get_time()
    elapsed = now - settings._previous_time
    settings._previous_time = now
    return elapsed


# TODO: Add docstrings
def print_passed_time():
    """TODO."""
    return _sec_to_str(get_passed_time())


# TODO: Finish docstrings
def timeout(func, args=(), timeout_duration=2, default=None, **kwargs):
    """Spwans thread and runs the given function using the args, kwargs, and return default value on timeout."""
    import threading

    class InterruptableThread(threading.Thread):
        def __init__(self):
            threading.Thread.__init__(self)
            self.result = default

        def run(self):
            self.result = func(*args, **kwargs)

    it = InterruptableThread()
    it.start()
    it.join(timeout_duration)
    return it.result


# TODO: Add docstrings
def get_latest_pypi_version():
    """TODO."""
    from subprocess import CalledProcessError, check_output

    try:  # needs to work offline as well
        result = check_output(["pip", "search", "scvelo"])
        return f"{result.split()[-1]}"[2:-1]
    except CalledProcessError:
        return "0.0.0"


# TODO: Add docstrings
def check_if_latest_version():
    """TODO."""
    from . import __version__

    latest_version = timeout(
        get_latest_pypi_version, timeout_duration=2, default="0.0.0"
    )
    if parse(__version__.rsplit(".dev")[0]) < parse(latest_version.rsplit(".dev")[0]):
        warn(
            "There is a newer scvelo version available on PyPI:\n",
            "Your version: \t\t",
            __version__,
            "\nLatest version: \t",
            latest_version,
        )
        
