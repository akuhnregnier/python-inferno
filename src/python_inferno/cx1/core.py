# -*- coding: utf-8 -*-
"""Run code on the cx1 cluster."""
import gc
import os
import random
import string
import sys
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from enum import Enum
from functools import partial
from pathlib import Path
from pprint import pformat, pprint
from subprocess import check_output

import cloudpickle
from jinja2 import Environment, FileSystemLoader
from loguru import logger
from wildfires.exceptions import NotCachedError

from ..exceptions import NoCX1Error
from ..utils import core_unpack_wrapped, tqdm

__all__ = (
    "get_cx1_dirs",
    "get_parsers",
    "run",
    "run_cx1",
)

template_dir = Path(__file__).resolve().parent / "templates"

# This syntax (using 2 lines) is required for multiprocessing (as it uses pickling).
Cached = Enum("Cached", ["CACHED", "UNCACHED"])
CACHED, UNCACHED = Cached


def batched_func_calls(func, batch_args, kwargs):
    out = []
    for single_args in zip(*batch_args):
        out.append(func(*single_args, **kwargs))
        gc.collect()
    return tuple(out)


def get_batch_args(args, batch_size):
    batch_args = []
    for i in range(0, len(args[0]), batch_size):
        batch = []
        for arg_values in args:
            batch.append(arg_values[i : i + batch_size])
        batch_args.append(batch)
    return tuple(batch_args)


def get_parsers():
    """Parse command line arguments to determine where to run a function."""
    parser = ArgumentParser(description="Run a function either locally or on CX1")

    parser.add_argument(
        "-v", "--verbose", action="store_true", help="output progress bar"
    )
    parser.add_argument(
        "-s", "--single", action="store_true", help="only run a single iteration"
    )
    parser.add_argument("--nargs", type=int, help="how many iterations to run")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=-1,
        help="how many iterations per batch",
    )
    parser.add_argument(
        "--uncached", action="store_true", help="only run uncached calls"
    )
    # Multiple threads / processes.
    parser.add_argument(
        "-n",
        "--n-cores",
        default=1,
        type=int,
        help="number of cores to use in parallel (default: single threaded)",
    )
    parser.add_argument("--info", action="store_true", help="print out info")
    mode_group = parser.add_mutually_exclusive_group(required=False)
    mode_group.add_argument(
        "--threads", action="store_true", help="use threads (default)"
    )
    mode_group.add_argument("--processes", action="store_true", help="use processes")

    subparsers = parser.add_subparsers(
        help="execution target", dest="dest", required=True
    )
    local_parser = subparsers.add_parser("local", help="run functions locally")
    cx1_parser = subparsers.add_parser("cx1", help="run functions on CX1 using PBS")
    check_parser = subparsers.add_parser(
        "check", help="locally check which calls are cached"
    )

    return dict(
        parser=parser,
        subparsers=subparsers,
        local_parser=local_parser,
        mode_group=mode_group,
        cx1_parser=cx1_parser,
        check_parser=check_parser,
    )


def check_in_store(func, *args, **kwargs):
    try:
        if hasattr(func, "check_in_store"):
            func.check_in_store(*args, **kwargs)
        else:
            func(*args, cache_check=True, **kwargs)
    except NotCachedError:
        return UNCACHED
    return CACHED


def check_local(func, args, kwargs, backend="threads", n_cores=1, verbose=False):
    chosen_executor = {"threads": ThreadPoolExecutor, "processes": ProcessPoolExecutor}[
        backend
    ]
    all_single_args = list(zip(*args))

    # Check which calls are not yet cached. This relies on functions implementing
    # the `cache_check` keyword argument if needed.
    checked = dict(present=[], uncached=[])
    uncached_args = []
    with chosen_executor(max_workers=n_cores) as executor:
        futures = []
        for single_args in all_single_args:
            futures.append(
                executor.submit(check_in_store, func, *single_args, **kwargs)
            )

        # Progress bar (out of order).
        for future in tqdm(
            as_completed(futures),
            desc="Checking",
            total=len(futures),
            disable=not verbose,
        ):
            # Ensure exceptions are caught here already.
            future.result()

        # Collect results in order.
        for single_args, future in zip(all_single_args, futures):
            if future.result() == CACHED:
                checked["present"].append((single_args, kwargs))
            else:
                checked["uncached"].append((single_args, kwargs))
                uncached_args.append(single_args)

    return checked, uncached_args


def run_local(func, batch_args, kwargs, backend="threads", n_cores=1, verbose=False):
    chosen_executor = {"threads": ThreadPoolExecutor, "processes": ProcessPoolExecutor}[
        backend
    ]
    out = []
    with chosen_executor(max_workers=n_cores) as executor:
        futures = []
        for single_batch_args in batch_args:
            futures.append(
                executor.submit(batched_func_calls, func, single_batch_args, kwargs)
            )

        # Progress bar (out of order).
        for future in tqdm(
            as_completed(futures),
            desc="Processing",
            total=len(futures),
            disable=not verbose,
        ):
            # Ensure exceptions are caught here already.
            future.result()

        # Collect results in order.
        for future in futures:
            out.extend(future.result())
    return tuple(out)


def get_cx1_dirs(job_name):
    # Store temporary files, e.g. different input arguments, in the EPHEMERAL
    # directory so the jobs can access them later.
    ephemeral = Path(os.environ["EPHEMERAL"])
    if not ephemeral.is_dir():
        raise RuntimeError(f"Ephemeral directory {ephemeral} was not found.")

    job_dir = ephemeral / job_name
    log_dir = job_dir / "pbs_output"

    # Create the necessary directories.
    log_dir.mkdir(parents=True, exist_ok=False)
    return job_dir, log_dir


def run_cx1(*, job_name, cmd, job_dir, log_dir, n_args, cx1_kwargs):
    if cx1_kwargs is False:
        raise NoCX1Error("`cx1_kwargs` is `False`, but running on CX1 was requested.")
    if cx1_kwargs is None:
        cx1_kwargs = {}

    # Submit either a single job (if there is only one set of arguments, or an
    # array job for multiple arguments.
    job_template = (
        "array_job_script.sh.jinja2" if n_args > 1 else "job_script.sh.jinja2"
    )

    # Render the job script template.
    if "walltime" not in cx1_kwargs:
        cx1_kwargs["walltime"] = "10:00:00"
    if "ncpus" not in cx1_kwargs:
        cx1_kwargs["ncpus"] = 1
    if "mem" not in cx1_kwargs:
        cx1_kwargs["mem"] = "5GB"

    job_template_kwargs = dict(
        job_name=job_name,
        cmd=cmd,
        job_log_dir=log_dir,
        step=1,
        min_index=0,
        max_index=n_args - 1,  # This is inclusive (PBS).
        **cx1_kwargs,
    )

    job_script = job_dir / "job_script.sh"
    with job_script.open("w") as f:
        f.write(
            Environment(loader=FileSystemLoader(template_dir))
            .get_template(job_template)
            .render(**job_template_kwargs)
        )

    # Submit the job.
    job_str = check_output(["qsub", str(job_script)]).decode().strip()
    logger.info(f"Submitted job '{job_str}' with job name '{job_name}'.")


def run_cx1_python(func, batch_args, kwargs, cx1_kwargs, verbose):
    job_name = f"{func.__name__}_{''.join(random.sample(string.ascii_lowercase, 8))}"
    job_dir, log_dir = get_cx1_dirs(job_name)

    # Store the function with arguments for later retrieval by the job.
    bound_functions = [
        partial(batched_func_calls, func, single_batch_args, kwargs)
        for single_batch_args in batch_args
    ]
    bound_functions_file = job_dir / "bound_functions.pkl"
    with bound_functions_file.open("wb") as f:
        cloudpickle.dump(bound_functions, f, -1)

    # Render the Python script template.
    python_template_kwargs = dict(
        bound_functions_file=bound_functions_file,
        pythonpath=repr(list(map(os.path.abspath, sys.path))),
    )

    python_script = job_dir / "python_script.py"
    with python_script.open("w") as f:
        f.write(
            Environment(loader=FileSystemLoader(template_dir))
            .get_template("python_script.py.jinja2")
            .render(**python_template_kwargs)
        )

    cmd = f"{sys.executable} {python_script}"

    run_cx1(
        job_name=job_name,
        cmd=cmd,
        job_dir=job_dir,
        log_dir=log_dir,
        n_args=len(batch_args),
        cx1_kwargs=cx1_kwargs,
    )


def run(
    func,
    *args,
    cx1_kwargs=None,
    get_parsers=get_parsers,
    return_local_args=False,
    batch_size=None,
    cmd_args=None,
    info=False,
    **kwargs,
):
    """Run a function depending on given (including command line) arguments.

    Command line arguments will dictate if this function is run locally or as an
    (array) job on the CX1 cluster.

    The function should cache/save its output internally.

    Note that `args` and `kwargs` will be pickled along with `func` itself in order to
    facilitate running as PBS jobs on CX1. It may be more efficient to defer loading
    of input data such that this is only carried out within `func` itself, based on
    the input arguments given here.

    For checking the presence of cached data, the given function should accept a
    `cache_check` keyword argument that will be set to True when checking is desired.

    Args:
        func (callable): Function to be run.
        *args (length-N iterables): Function arguments. These will be zipped before
            being passed to the function.
        cx1_kwargs: Further specification of the type of job that will be used to
            run the function on the cluster. The following arguments are supported:
            'walltime', 'ncpus', and 'mem'. If `False` is given, trying to run on cx1
            will raise an error.
        get_parsers (callable): Callable that returns a dictionary containing (at
            least) a 'parser' key, which references the parser object used to parse
            command line arguments. This return value should support the following
            call (i.e. `get_parsers()['parser'].parse_args()`).
        return_local_args (bool):
            If True, return the arguments and kwargs along with the results (only
            applies for 'local').
        batch_size (int): How many samples per iteration.
        cmd_args (object): Namespace-like object which may be given to circumvent
            using `sys.argv` in combination with the standard parser (i.e. if
            `cmd_args` is given, `get_parsers` is not used).
        info (bool): If True, print out information about the run to stdout and return.
        **kwargs: Function keyword arguments. These will be given identically to each
            function call, as opposed to `args`.

    Returns:
        tuple or None: The output results are returned if running locally. Otherwise,
            None is returned.

    Raises:
        NoCX1Error: If cx1_kwargs is `False` but running on CX1 was requested.

    """
    if cmd_args is None:
        cmd_args = get_parsers()["parser"].parse_args()
    info = cmd_args.info
    if info:
        print(f"Number of args: {len(list(zip(*args)))}")
        return
    verbose = cmd_args.verbose
    single = cmd_args.single
    nargs = cmd_args.nargs
    if cmd_args.batch_size != -1 and batch_size is not None:
        raise ValueError(
            "Cannot give both '--batch-size' argument and batch_size keyword."
        )

    if cmd_args.batch_size != -1:
        # Use the command-line arg if given.
        batch_size = cmd_args.batch_size
    elif batch_size is None:
        # Default value of 1.
        batch_size = 1

    kwargs = {**dict(single=single, nargs=nargs, verbose=verbose), **kwargs}

    if len(args) == 0 or len(args[0]) == 0:
        logger.warning("No args given.")
        return

    if single:
        # Only run a single iteration.
        args = list(args)
        args[0] = args[0][:1]  # Because zip() is used later on, this is sufficient.
        args = tuple(args)
    elif nargs:
        # Only run a limited number of iterations.
        args = list(args)
        args[0] = args[0][:nargs]  # Because zip() is used later on, this is sufficient.
        args = tuple(args)

    if cmd_args.dest == "check" or cmd_args.uncached:
        checked, uncached_args = check_local(
            func=func,
            args=args,
            kwargs=kwargs,
            backend=("processes" if cmd_args.processes else "threads"),
            n_cores=cmd_args.n_cores,
            verbose=verbose,
        )

        pprint({key: len(val) for key, val in checked.items()})

        if cmd_args.dest == "check":
            # Only checking was requested.
            logger.info(f"Cache status:\n{pformat(checked)}")
            logger.info(
                f"Number of uncached calls: {len(uncached_args)}/{len(args[0])}"
            )
            return checked
        # Otherwise, we want to run only the uncached calls.
        args = tuple(zip(*uncached_args))

    # Batch the input arguments.
    batch_args = get_batch_args(args, batch_size)

    if cmd_args.dest == "local":
        raw_out = run_local(
            func=func,
            batch_args=batch_args,
            kwargs=kwargs,
            backend=("processes" if cmd_args.processes else "threads"),
            n_cores=cmd_args.n_cores,
            verbose=verbose,
        )
        out = core_unpack_wrapped(*raw_out)
        if len(raw_out) == 1:
            out = (out,)

        if return_local_args:
            return args, kwargs, out
        return out
    elif cmd_args.dest == "cx1":
        run_cx1_python(
            func=func,
            batch_args=batch_args,
            kwargs=kwargs,
            cx1_kwargs=cx1_kwargs,
            verbose=verbose,
        )
