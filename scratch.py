import pandas as pd
import numpy as np
import os

# set path
root = 'test'
img_dir = 'img/no_deck/'
item_img_dir = os.path.join(root, img_dir)

# read all jpg images in item_img_dir
img_fnames = [f for f in os.listdir(item_img_dir) if f.endswith('.jpg')]
img_paths = [img_dir + f for f in img_fnames]
print(img_paths)

import random
random.randint(0, 100)

x = 1
y = 2
x, y  = y , x
print(x, y)

random()
