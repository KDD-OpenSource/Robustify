import time
import multiprocessing as mp
from src.utils.utils import exec_cfg
from src.utils.utils import load_cfgs


import torch.nn as nn
import torch


def main():

    parallel = False
    cfgs = load_cfgs()
    start_timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")
    if parallel:
        pool = mp.Pool(int(1 * mp.cpu_count()))
        for cfg in cfgs:
            pool.apply_async(exec_cfg, args=((cfg, start_timestamp)))
        pool.close()
        pool.join()

    else:
        for cfg in cfgs:
            exec_cfg(cfg, start_timestamp)
#
#
if __name__ == "__main__":
    main()
