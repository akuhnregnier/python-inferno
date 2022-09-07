# -*- coding: utf-8 -*-
"""

Note that this should only be run in a single thread at a time!
I.e. run the cached functions separately before calling them in a parallel context
(cache retrieval is fine in multiple processes/threads).

"""
import os
import shlex
import shutil
import subprocess
from itertools import product
from multiprocessing import Process
from pathlib import Path
from tempfile import NamedTemporaryFile

import dask.array as da
import iris
import numpy as np
import pandas as pd
from loguru import logger
from numba import prange
from wildfires.utils import parallel_njit

from .cache import cache, mark_dependency

pnv_tif_1km = (
    Path("~/tmp/pnv_map_1km/")
    / "pnv_biome.type_biome00k_c_1km_s0..0cm_2000..2017_v0.1.tif"
).expanduser()

pnv_csv_file = (
    Path("~/tmp/pnv_map_1km")
    / "pnv_biome.type_biome00k_c_1km_s0..0cm_2000..2017_v0.1.tif.csv"
)
GDAL_BIN = str(
    Path(os.environ["PYENV_ROOT"])
    / "versions/miniconda3-latest/envs/python-inferno/bin"
)


@mark_dependency
def gdal_regrid_to_N96e_mode(*, in_file, out_file):
    """Used to regrid mega biomes to N96e, e.g. from 1km resolution.

    Note: '-r mode' is used to output the most common mega biome in each new grid
    cell.

    """
    cmd = shlex.split(
        f"""{GDAL_BIN}/gdalwarp -t_srs EPSG:4626 -r mode -te -180 -90 180 90
        -ts 192 144 -of netCDF -multi -wo NUM_THREADS=10 -wm 9000
        --config GDAL_CACHEMAX 1024
        {in_file} {out_file}"""
    )
    out = subprocess.run(cmd)
    assert out.returncode == 0


@mark_dependency
def gdal_to_nc(*, in_file, out_file):
    cmd = shlex.split(f"{GDAL_BIN}/gdal_translate -of netCDF {in_file} {out_file}")
    out = subprocess.run(cmd)
    assert out.returncode == 0


@mark_dependency
def get_mega_pnv_biome_map():
    pnv_df = pd.read_csv(pnv_csv_file, header=0, index_col=0)

    # Create another cube analogous to the above, but with PNV types aggregated
    # across the 'Mega_biome_classification' groupings.
    mega_biomes = np.unique(pnv_df["Mega_biome_classification"])
    mega_region_values = np.arange(1, 1 + mega_biomes.size)
    assert mega_biomes.size == mega_region_values.size
    mega_biome_name_map = {
        key: value for key, value in zip(mega_biomes, mega_region_values)
    }

    # Mapping array from original PNV to Mega PNV.
    mega_region_map = np.zeros((pnv_df["Number"].max() + 1,), dtype=np.uint8)
    for number in pnv_df["Number"]:
        matching_mega_biome_names = pnv_df[pnv_df["Number"] == number][
            "Mega_biome_classification"
        ].values
        assert (
            matching_mega_biome_names.size == 1
        ), "There should only be 1 match for each number."
        matching_mega_biome_number = mega_biome_name_map[matching_mega_biome_names[0]]

        mega_region_map[number] = matching_mega_biome_number

    # +1 to account for the 0 'fill values'.
    assert len(np.unique(mega_region_map)) == mega_biomes.size + 1

    return mega_region_map


@parallel_njit
@mark_dependency
def _mega_biome_mapping(normal_biomes, mapping_arr):
    mega_biomes = np.zeros_like(normal_biomes, dtype=np.uint8)

    indices = list(np.ndindex(normal_biomes.shape))
    for i in prange(len(indices)):
        index = indices[i]

        # Simply propagate the fill value (for invalid values).
        if normal_biomes[index] == 255:
            mega_biomes[index] = 255
        else:
            # Apply mapping from PNV to Mega regions.
            mega_biomes[index] = mapping_arr[normal_biomes[index]]

    return mega_biomes


@mark_dependency
def mega_biome_mapping(normal_biomes, mapping_arr):
    assert np.ma.isMaskedArray(normal_biomes)
    mega_biomes = _mega_biome_mapping(normal_biomes.data, mapping_arr)
    assert (
        np.sum(mega_biomes == 0) == 0
    ), "All locations should be masked or a valid mega biome, i.e. >=1"
    return np.ma.MaskedArray(mega_biomes, mask=normal_biomes.mask)


@mark_dependency
def dask_pnv_mega_biomes(*, in_file, out_file):
    mega_pnv_biome_map = get_mega_pnv_biome_map()

    cube = iris.load_cube(str(in_file))

    mega_cube_data = da.map_blocks(
        mega_biome_mapping,
        cube.core_data(),
        mapping_arr=mega_pnv_biome_map,
        meta=np.array([], dtype=np.uint8),
        dtype=np.uint8,
    )

    mega_cube = cube.copy(data=mega_cube_data)

    with NamedTemporaryFile(
        mode="w", prefix="mega_cube", suffix=".nc", delete=False
    ) as tmp_file:
        iris.save(mega_cube, tmp_file.name, fill_value=255)

    shutil.move(tmp_file.name, str(out_file))


@cache(
    dependencies=[
        gdal_regrid_to_N96e_mode,
        gdal_to_nc,
        get_mega_pnv_biome_map,
        _mega_biome_mapping,
        mega_biome_mapping,
        dask_pnv_mega_biomes,
    ]
)
def get_pnv_mega_regions():
    """Get a cube that represents PNV regions aggregated to MEGA biomes.

    See also:
     - https://github.com/Envirometrix/PNVmaps

    The cube will have `attributes` akin to:
        >>> # {
        >>> #     "regions": {
        >>> #         0: "Ocean",
        >>> #         1: "BONA (Boreal North America)",
        >>> #         14: "AUST (Australia and New Zealand)",
        >>> #     },
        >>> #     "short_regions": {0: "Ocean", 14: "AUST"},
        >>> #     "region_codes": {"Ocean": 0, "BONA": 1, "AUST": 14},
        >>> # }

    """
    if not pnv_tif_1km.is_file():
        raise ValueError(f"Source tif file: {pnv_tif_1km}, does not exist.")

    logger.info(f"Using source tif file: {pnv_tif_1km}.")

    pnv_nc_1km = (
        Path("~/tmp/pnv_map_1km/processed")
        / "pnv_biome.type_biome00k_c_1km_s0..0cm_2000..2017_v0.1.nc"
    ).expanduser()

    if pnv_nc_1km.is_file():
        logger.info(f"nc_1km file: '{pnv_nc_1km}' already exists, deleting.")
        pnv_nc_1km.unlink()
    else:
        pnv_nc_1km.parent.mkdir(exist_ok=True, parents=False)

    # Create netCDF version of tif file.
    gdal_to_nc(in_file=pnv_tif_1km, out_file=pnv_nc_1km)

    mega_nc_1km = pnv_nc_1km.with_name("mega_biome_1km.nc")

    # Map PNV biomes to mega biome regions.
    logger.info("Aggregating MEGA regions.")
    # NOTE This is started in a new process because it otherwise interferes with the
    # next GDAL command. It is currently unclear why.
    p = Process(
        target=dask_pnv_mega_biomes,
        kwargs=dict(in_file=pnv_nc_1km, out_file=mega_nc_1km),
    )
    p.start()
    p.join()
    logger.info("Done aggregating MEGA regions.")

    regrid_out_file = mega_nc_1km.with_name("mega_biome_n96e.nc")
    if regrid_out_file.is_file():
        regrid_out_file.unlink()

    # Regrid 1km netCDF data to N96e netCDF data.
    logger.info(f"Regridding to {regrid_out_file}")
    gdal_regrid_to_N96e_mode(in_file=mega_nc_1km, out_file=regrid_out_file)
    logger.info("Done regridding.")

    pnv_df = pd.read_csv(pnv_csv_file, header=0, index_col=0)

    mega_biomes = np.unique(pnv_df["Mega_biome_classification"])
    mega_region_values = np.arange(1, 1 + mega_biomes.size)
    mega_region_sources = {}
    for mega_biome in mega_biomes:
        mega_region_sources[mega_biome] = pnv_df[
            pnv_df["Mega_biome_classification"] == mega_biome
        ]["Number"].values

    assert len(mega_biomes) == len(mega_region_sources) == len(mega_region_values)

    mega_pnv_cube = iris.load_cube(str(regrid_out_file))
    mega_pnv_cube.attributes["regions"] = dict(zip(mega_region_values, mega_biomes))
    mega_pnv_cube.attributes["short_regions"] = dict(
        zip(mega_region_values, mega_biomes)
    )
    mega_pnv_cube.attributes["regions_codes"] = {
        value: key for key, value in mega_pnv_cube.attributes["short_regions"].items()
    }

    # Combine tundra and dry tundra regions.
    for key, index in product(("regions", "short_regions"), (3, 8)):
        del mega_pnv_cube.attributes[key][index]
    for name in ("tundra", "dry tundra"):
        del mega_pnv_cube.attributes["regions_codes"][name]

    # The new '(dry) tundra' region will have the code 10.
    mega_pnv_cube.attributes["regions"][10] = "tundra"
    mega_pnv_cube.attributes["short_regions"][10] = "tundra"
    mega_pnv_cube.attributes["regions_codes"]["tundra"] = 10

    mega_pnv_cube.data[(mega_pnv_cube.data == 3) | (mega_pnv_cube.data == 8)] = 10

    return mega_pnv_cube
