import dask
import dask_awkward as dak
import awkward as ak
import numpy as np
from coffea import dataset_tools
from coffea.nanoevents import NanoEventsFactory, PFNanoAODSchema
from ndcctools.taskvine import DaskVine
import fastjet
import time
import os
import warnings
import scipy
import cloudpickle
import inspect
import dask_awkward.lib.structure

full_start = time.time()


def compute(hlg, keys):
    print('start compute')

    m = DaskVine(
        [9123, 9128],
        name=f"{os.environ['USER']}-hgg2",
    )

    m.tune("transfer-temps-recovery", 1)

    computed = m.get(hlg, keys, collapse_hlg=False, resources={"cores": 1}, env_vars={'PATH': '/scratch365/jzhou24/env/bin/:$PATH'})

    full_stop = time.time()
    print('full run time is ' + str((full_stop - full_start) / 60))


if __name__ == "__main__":

    with open('keys.pkl', 'rb') as f:
        keys = cloudpickle.load(f)

    with open('expanded_hlg.pkl', 'rb') as f:
        hlg = cloudpickle.load(f)


    for k, v in hlg.items():
        if isinstance(v, str):
            pass
            # print(v)
        elif isinstance(v, tuple):
            print(v[0], type(v[0]))
        else:
            print(f"error: {k}")
            exit(1)

    compute(hlg, keys)