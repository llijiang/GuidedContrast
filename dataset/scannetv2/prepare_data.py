import plyfile
import numpy as np
import multiprocessing as mp
import torch
import argparse
import os

# Map relevant classes to {0,1,...,19}, and ignored classes to -100
remapper = np.ones(150) * (-100)
for i, x in enumerate([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]):
    remapper[x] = i

parser = argparse.ArgumentParser()
parser.add_argument('--split', help='data split (train/val/test)', default='train')
parser.add_argument('--scannet_path', default='/path/to/ScanNet')
opt = parser.parse_args()

split = opt.split
print(f'data split: {split}')
os.makedirs(split, exist_ok=True)

split_map = {'train': 'scans', 'val': 'scans', 'test': 'scans_test'}
split_path = os.path.join(opt.scannet_path, split_map[split])
split_fns = open(f'scannetv2_split/scannetv2_{split}.txt').readlines()
split_fns = [i.strip() for i in split_fns]

files = sorted([os.path.join(split_path, i, i + '_vh_clean_2.ply') for i in split_fns])

def f(fn):
    print(fn)

    f = plyfile.PlyData().read(fn)
    points = np.array([list(x) for x in f.elements[0]])
    coords = np.ascontiguousarray(points[:, :3] - points[:, :3].mean(0))
    colors = np.ascontiguousarray(points[:, 3:6]) / 127.5 - 1
    out = (coords, colors)

    if split != 'test':
        fn2 = fn[:-3] + 'labels.ply'
        f2 = plyfile.PlyData().read(fn2)
        sem_labels = remapper[np.array(f2.elements[0]['label'])]
        out = (coords, colors, sem_labels)

    torch.save(out, os.path.join(split, fn[:-15].split('/')[-1]+'.pth'))
    print('Saving to ' + os.path.join(split, fn[:-15].split('/')[-1]+'.pth'))

p = mp.Pool(processes=mp.cpu_count())
p.map(f, files)
p.close()
p.join()
