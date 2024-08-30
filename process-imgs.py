import pandas as pd
import numpy as np
import os 

# set path 
root = 'test'
img_dir = 'img/no_deck/'
item_img_dir = os.path.join(root, img_dir)

# read all jpg images in item_img_dir
img_fnames = [f for f in os.listdir(item_img_dir) if f.endswith('.jpg')]
# img_paths = [img_dir + f for f in img_fnames]

# print(len(img_paths))
# print(img_paths)

n_imgs = 665 
img_paths = [f'img/no_deck/object{i}.jpg' for i in range(1, n_imgs+1)]

# make a csv file such that there are two columns with headers 'image_l' and 'image_r'. 
# below 'image_l', lists all even indexed images and below 'image_r', lists all odd indexed images
df = pd.DataFrame()
n_rows = 332
value_set = [0, 40, 60, 100]
df['image_l'] = img_paths[::2][:n_rows]
df['image_r'] = img_paths[1::2][:n_rows]
# add a column 'value_l', the value for each row is a random sample from value_set
df['value_l'] = np.random.choice(value_set, n_rows)
# add a column 'value_r', the value for each row is a random sample from value_set but different from 'value_l'
value_r = np.random.choice(value_set, n_rows)
for i in range(n_rows):
    while value_r[i] == df['value_l'][i]:
        value_r[i] = np.random.choice(value_set)
df['value_r'] = value_r
print(df.head())

df.to_csv('test/trials-info-test.csv', index=False)


# make a csv file such that there is a single column with headers 'image', and list all items in img_paths
df = pd.DataFrame()
df['image'] = img_paths
# add a column 'value', the value for each row is a random sample from value_set
# df['value'] = np.random.choice(value_set, len(img_paths))
df['value'] = np.arange(len(img_paths)) + 1 
print(df.head())

df.to_csv('test/trials-info-single-col-test.csv', index=False)

# load a csv file 
trial_info_single = pd.read_csv('test/trials-info-single-col-test.csv')