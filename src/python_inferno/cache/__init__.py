# -*- coding: utf-8 -*-
import os
from functools import partial
from pathlib import Path

from wildfires.cache import (
    IN_STORE,
    ProxyMemory,
    cache_hash_value,
    mark_dependency,
    process_proxy,
)

memory = ProxyMemory(
    Path(os.environ["EPHEMERAL"]) / "python_inferno" / "joblib_cache",
    verbose=0,
    # Prevent caching in FIREDATA.
    root_dir="",
)
cache = memory.cache

process_proxy = partial(process_proxy, memory=memory)
cache_hash_value = partial(cache_hash_value, hash_func=memory.get_hash)
