import glob
import numpy as np
import multiprocessing as mp
import torch
import os
import argparse

semantic_names = ['ceiling', 'floor', 'wall', 'beam', 'column', 'window', 'door', 'chair', 'table', 'bookcase', 'sofa', 'board', 'clutter']
semantic_name2id = {}
for i, names in enumerate(semantic_names):
    semantic_name2id[names] = i

parser = argparse.ArgumentParser()
parser.add_argument('--s3dis_path', default='/path/to/S3DIS')
opt = parser.parse_args()

split = os.path.join(opt.s3dis_path, 'Area_[1-6]')
save_root = 's3dis'
if not os.path.exists(save_root):
    os.makedirs(save_root)
files = sorted(glob.glob(split + '/*'))

def f(fn):
    print(fn)

    coords = []
    colors = []
    semantic_labels = []
    instance_files = sorted(glob.glob(os.path.join(fn, 'Annotations', '*.txt')))

    for i, ins_fn in enumerate(instance_files):
        class_name = ins_fn.split('/')[-1].split('.')[0].split('_')[0]
        if class_name in semantic_names:
            class_id = semantic_name2id[class_name]
        else:
            class_id = -100

        v = np.loadtxt(ins_fn)   # Area_5/hallway_6/Annotations/ceiling_1.txt Line 180389

        coords.append(v[:, :3])
        colors.append(v[:, 3:6])
        semantic_labels.append(np.ones(v.shape[0]) * class_id)

    coords = np.concatenate(coords).astype(np.float32)
    colors = np.concatenate(colors).astype(np.float32)
    semantic_labels = np.concatenate(semantic_labels).astype(np.float32)

    coords = np.ascontiguousarray(coords - coords.mean(0))
    colors = np.ascontiguousarray(colors) / 127.5 - 1

    save_path = os.path.join(save_root, fn.split('/')[-2] + '_' + fn.split('/')[-1] + '.pth')
    torch.save((coords, colors, semantic_labels), save_path)
    print('Saving to ' + save_path)

p = mp.Pool(processes=mp.cpu_count())
p.map(f, files)
p.close()
p.join()
