import os
import time


rounds=900

import sys
mn=0
mx=900
if len(sys.argv)>2:
    mn,mx=int(sys.argv[1]),int(sys.argv[2])
elif len(sys.argv)>1:
    mx=int(sys.argv[1])


for i in range(mn,mx):
    os.system(f"python3 base.py {i}")
    time.sleep(1)

