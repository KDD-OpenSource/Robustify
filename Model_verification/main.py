import time
import multiprocessing as mp
from src.utils.util_main import exec_cfg
from src.utils.util_main import load_cfgs


import torch.nn as nn
import torch


def main():
    # executes different cfg-files in parallel
    parallel = False
    cpu_fraction = 1
    cfgs = load_cfgs()
    start_timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")
    if parallel:
        pool = mp.Pool(int(cpu_fraction * mp.cpu_count()))
        for cfg in cfgs:
            pool.apply_async(exec_cfg, args=((cfg, start_timestamp)))
        pool.close()
        pool.join()
    else:
        for cfg in cfgs:
            exec_cfg(cfg, start_timestamp)


if __name__ == "__main__":
    main()
