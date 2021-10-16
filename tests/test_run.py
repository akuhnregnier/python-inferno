# -*- coding: utf-8 -*-
import sys
import threading
from pathlib import Path
from subprocess import check_call
from tempfile import TemporaryDirectory
from time import sleep

from python_inferno.cx1 import run


def get_cmd_args(dest="local", batch_size=-1, n_cores=1):
    return type(
        "cmd_args",
        (object,),
        dict(
            verbose=False,
            single=False,
            nargs=None,
            batch_size=batch_size,
            dest=dest,
            uncached=False,
            processes=False,
            threads=False,
            info=False,
            n_cores=n_cores,
        ),
    )


def test_run():
    def f(x, **kwargs):
        return x + 1

    assert run(f, [1], cmd_args=get_cmd_args()) == (2,)
    assert run(f, [1, 2], cmd_args=get_cmd_args()) == (2, 3)


def test_batched():
    def get_thread_id(*args, **kwargs):
        # Do some 'work' to prevent single threads from taking on more than a single
        # call.
        sleep(0.05)
        return threading.get_ident()

    # Without multi-threading, this should just repeatedly return the same thread ID.
    assert len(set(run(get_thread_id, [None] * 3, cmd_args=get_cmd_args()))) == 1

    # With multi-threading, this should return the different threads' IDs.
    assert (
        len(set(run(get_thread_id, [None] * 3, cmd_args=get_cmd_args(n_cores=3)))) == 3
    )

    # In batches of 2, only 2 'batched' calls should be made (2 & 1 calls per batch).
    assert (
        len(
            set(
                run(
                    get_thread_id,
                    [None] * 3,
                    cmd_args=get_cmd_args(batch_size=2, n_cores=3),
                )
            )
        )
        == 2
    )

    # In batches of 3, only 1 'batched' call should be made, since only 3 calls are
    # made in total.
    assert (
        len(
            set(
                run(
                    get_thread_id,
                    [None] * 3,
                    cmd_args=get_cmd_args(batch_size=3, n_cores=3),
                )
            )
        )
        == 1
    )


def test_batched_alt_call():
    """Similar tests to above but with an alternate calling convention for batching."""

    def get_thread_id(*args, **kwargs):
        # Do some 'work' to prevent single threads from taking on more than a single
        # call.
        sleep(0.05)
        return threading.get_ident()

    # In batches of 2, only 2 'batched' calls should be made (2 & 1 calls per batch).
    assert (
        len(
            set(
                run(
                    get_thread_id,
                    [None] * 3,
                    cmd_args=get_cmd_args(batch_size=-1, n_cores=3),
                    batch_size=2,
                )
            )
        )
        == 2
    )

    # In batches of 3, only 1 'batched' call should be made, since only 3 calls are
    # made in total.
    assert (
        len(
            set(
                run(
                    get_thread_id,
                    [None] * 3,
                    cmd_args=get_cmd_args(batch_size=-1, n_cores=3),
                    batch_size=3,
                )
            )
        )
        == 1
    )


def test_cmd_line_run():
    script = """
from python_inferno.cx1 import run

def f(x, **kwargs):
    return x + 1

assert run(f, [1]) == (2,)
"""
    # Write this script to a temporary file and run it.

    with TemporaryDirectory() as tempdir:
        fpath = Path(tempdir) / "test.py"
        with fpath.open("w") as f:
            f.write(script)
        check_call([sys.executable, str(fpath), "local"])
