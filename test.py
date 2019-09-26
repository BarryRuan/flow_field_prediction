import numpy as np
import argparse
import math
import os
from utils import get
from utils import *
import utils

v_ff = []
for i in range(1, 3):
    print('read', str(i))
    v_ff.append(utils.read_flow_field('data/velocity/%d' % (i)))

v_ff = np.array(v_ff)

print(np.size(v_ff))
print(v_ff[1, 2:5, 1:8, :])
