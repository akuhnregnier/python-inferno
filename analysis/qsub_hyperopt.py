#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import random
import string
from argparse import ArgumentParser

from python_inferno.cx1 import get_cx1_dirs, run_cx1

if __name__ == "__main__":
    parser = ArgumentParser(description="Run hyperopt workers on CX1.")
    parser.add_argument("-n", help="how many workers to run", type=int, default=1)

    args = parser.parse_args()

    job_name = f"hyperopt_{''.join(random.sample(string.ascii_lowercase, 8))}"
    job_dir, log_dir = get_cx1_dirs(job_name)

    run_cx1(
        job_name=job_name,
        cmd="/rds/general/user/ahk114/home/.pyenv/versions/miniconda3-latest/envs/wildfires/bin/hyperopt-mongo-worker --mongo=maritimus.webredirect.org:1234/ba --poll-interval=10",
        job_dir=job_dir,
        log_dir=log_dir,
        n_args=args.n,
        cx1_kwargs=dict(walltime="24:00:00", ncpus=2, mem="28GB"),
    )
