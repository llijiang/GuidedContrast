import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import SharedArray as SA
import os, sys, glob
sys.path.append('../')

from lib.ops import voxelization_idx
from util.pointdata_process import dataAugment, elastic, get_overlap_xy_crops, get_xy_crop
from util import utils


class ScanNet(Dataset):
    def __init__(self, cfg, file_names, aug_options, test=False, unlabeled=False):
        super(ScanNet, self).__init__()

        self.cache = cfg.cache
        self.dist = cfg.dist
        self.local_rank = cfg.local_rank

        self.scale = cfg.scale

        self.file_names = file_names
        if not self.cache:
            self.files = [torch.load(i) for i in self.file_names]
        else:
            num_gpus = 1 if not self.dist else torch.cuda.device_count()
            rk = self.local_rank % num_gpus
            utils.create_shared_memory(file_names[rk::num_gpus])

        self.jit, self.flip, self.rot, self.elastic, self.crop, self.rgb_aug = aug_options
        self.flipv, self.rotv = None, None

        self.unlabeled = unlabeled
        self.crop_size = cfg.crop_size
        self.crop_max_iters = cfg.crop_max_iters

        self.test = test

    def item_process(self, xyz_origin, rgb, label=None):
        xyz = dataAugment(xyz_origin, self.jit, self.flip, self.rot, flipv=self.flipv, rotv=self.rotv)
        xyz = xyz * self.scale
        if self.elastic:
            xyz = elastic(xyz, 6 * self.scale // 50, 40 * self.scale / 50)
            xyz = elastic(xyz, 20 * self.scale // 50, 160 * self.scale / 50)
        xyz -= xyz.min(0)

        xyz_float = xyz / self.scale

        if self.rgb_aug:
            rgb += (np.random.randn(3) * 0.1)

        item = {'xyz': xyz, 'xyz_float': xyz_float, 'rgb': rgb}
        if label is not None:
            item['label'] = label

        return item

    def __getitem__(self, id):
        fn = self.file_names[id].split('/')[-1].split('.')[0]
        if self.cache:
            xyz_origin = SA.attach(f'shm://{fn}_xyz').copy()
            rgb = SA.attach(f'shm://{fn}_rgb').copy()
            if not self.test: label = SA.attach(f'shm://{fn}_label').copy()
        else:
            xyz_origin = self.files[id][0]
            rgb = self.files[id][1]
            if not self.test: label = self.files[id][2]

        if self.unlabeled:
            # get overlapped crops
            idx1, idx2, idxo, idxo1, idxo2 = get_overlap_xy_crops(xyz_origin, self.crop_size, self.crop_max_iters)

            xyz_origin1 = xyz_origin[idx1].copy()
            xyz_origin2 = xyz_origin[idx2].copy()
            rgb1 = rgb[idx1].copy()
            rgb2 = rgb[idx2].copy()
            if not self.test:
                label1 = label[idx1].copy()
                label2 = label[idx2].copy()

            # process
            item1 = self.item_process(xyz_origin1, rgb1, label=None if self.test else label1)
            item2 = self.item_process(xyz_origin2, rgb2, label=None if self.test else label2)

            item = {'idx': (idx1, idx2), 'idxo': (idxo1, idxo2)}
            for k in item1.keys():
                item[k] = (item1[k], item2[k])

        else:
            # get crop
            if self.crop:
                idx = get_xy_crop(xyz_origin, self.crop_size)

                xyz_origin = xyz_origin[idx]
                rgb = rgb[idx]
                if not self.test:
                    label = label[idx]

            # process
            item = self.item_process(xyz_origin, rgb, label=None if self.test else label)

        item['item_id'] = id
        item['item_fn'] = fn

        return item

    def __len__(self):
        return len(self.file_names)


class MyDataset:
    def __init__(self, cfg, test=False):
        self.cfg = cfg

        self.data_root = cfg.data_root
        self.dataset = cfg.dataset

        self.batch_size = cfg.batch_size
        self.train_workers = cfg.train_workers
        self.val_workers = cfg.train_workers

        self.dist = cfg.dist

        self.train_flip = cfg.train_flip
        self.train_rot = cfg.train_rot
        self.train_jit = cfg.train_jit
        self.train_elas = cfg.train_elas

        self.full_scale = cfg.full_scale

        self.labeled_ratio = cfg.labeled_ratio

        if test:
            self.test_split = cfg.split  # val or test
            self.test_workers = cfg.test_workers

    def trainLoader(self):
        self.train_file_names = sorted(glob.glob(os.path.join(self.data_root, self.dataset, 'train', '*.pth')))

        if self.labeled_ratio < 1:
            split_path = os.path.join(self.data_root, self.dataset, 'data_split')
            split_fn = os.path.join(split_path, f'{int(self.labeled_ratio * 100)}.txt')
            with open(split_fn) as f:
                l_fns_ = f.readlines()
                l_fns_ = [i.strip() for i in l_fns_]
            self.labeled_fns = [fn for fn in self.train_file_names if fn.split('/')[-1].split('.')[0][:12] in l_fns_]
            self.unlabeled_fns = [fn for fn in self.train_file_names if
                                  fn.split('/')[-1].split('.')[0][:12] not in l_fns_]
        else:
            self.labeled_fns, self.unlabeled_fns = self.train_file_names, self.train_file_names

        l_train_set = ScanNet(
            self.cfg, self.labeled_fns,
            [self.train_jit, self.train_flip, self.train_rot, self.train_elas, True, True],
            # [jit, flip, rot, elastic, crop, rgb_aug]
            unlabeled=False
        )

        u_train_set = ScanNet(
            self.cfg, self.unlabeled_fns,
            [self.train_jit, self.train_flip, self.train_rot, self.train_elas, True, True],
            unlabeled=True
        )

        self.l_train_sampler = torch.utils.data.distributed.DistributedSampler(
            l_train_set, shuffle=True, drop_last=False) if self.dist else None
        self.l_train_data_loader = DataLoader(l_train_set, batch_size=self.batch_size, collate_fn=self.get_batch_data,
                                              num_workers=self.train_workers, shuffle=(self.l_train_sampler is None),
                                              sampler=self.l_train_sampler, drop_last=False, pin_memory=True,
                                              worker_init_fn=self._worker_init_fn_)

        self.u_train_sampler = torch.utils.data.distributed.DistributedSampler(
            u_train_set, shuffle=True, drop_last=False) if self.dist else None
        self.u_train_data_loader = DataLoader(u_train_set, batch_size=self.batch_size, collate_fn=self.get_batch_data,
                                              num_workers=self.train_workers, shuffle=(self.u_train_sampler is None),
                                              sampler=self.u_train_sampler, drop_last=False, pin_memory=True,
                                              worker_init_fn=self._worker_init_fn_)

    def valLoader(self):
        self.val_file_names = sorted(glob.glob(os.path.join(self.data_root, self.dataset, 'val', '*.pth')))
        val_set = ScanNet(
            self.cfg, self.val_file_names,
            [False, False, False, False, False, False],
            unlabeled=False
        )
        self.val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_set, shuffle=False, drop_last=False) if self.dist else None
        self.val_data_loader = DataLoader(val_set, batch_size=self.batch_size, collate_fn=self.get_batch_data,
                                          num_workers=self.val_workers, shuffle=False, sampler=self.val_sampler,
                                          drop_last=False, pin_memory=True, worker_init_fn=self._worker_init_fn_)

    def testLoader(self):
        assert self.test_split in ['val', 'test']
        self.test_file_names = sorted(glob.glob(os.path.join(self.data_root, self.dataset, self.test_split, '*.pth')))
        self.test_set = ScanNet(
            self.cfg, self.test_file_names,
            [False, True, True, False, False, False],
            test=(self.test_split == 'test'),
            unlabeled=False
        )
        self.test_sampler = torch.utils.data.distributed.DistributedSampler(
            self.test_set, shuffle=False, drop_last=False) if self.dist else None
        self.test_data_loader = DataLoader(self.test_set, batch_size=self.batch_size, collate_fn=self.get_batch_data,
                                           num_workers=self.test_workers, shuffle=False, sampler=self.test_sampler,
                                           drop_last=False, pin_memory=True, worker_init_fn=self._worker_init_fn_)

    def _worker_init_fn_(self, worker_id):
        torch_seed = torch.initial_seed()
        np_seed = torch_seed % 2 ** 32 - 1
        np.random.seed(np_seed)

    def get_batch_data(self, item_list):
        def get_value_list(item_list, key):
            values = []
            for i, item in enumerate(item_list):
                value = item[key]
                if isinstance(value, tuple):
                    values.extend(list(value))
                else: values.append(value)
            return values

        def cat_value_list(values, dtype=None, add_batch_idx=False):
            for i, value in enumerate(values):
                value = torch.from_numpy(value)
                if dtype: value = value.to(dtype)
                if add_batch_idx:
                    value = torch.cat([torch.ones(value.shape[0], 1).to(value) * i, value], 1)
                values[i] = value
            values = torch.cat(values, 0)
            return values

        test = not ('label' in item_list[0])
        unlabeled = 'idxo' in item_list[0]

        locs = cat_value_list(get_value_list(item_list, 'xyz'), dtype=torch.long, add_batch_idx=True)
        locs_float = cat_value_list(get_value_list(item_list, 'xyz_float'), dtype=torch.float32)
        feats = cat_value_list(get_value_list(item_list, 'rgb'))
        if not test:
            labels = cat_value_list(get_value_list(item_list, 'label'), dtype=torch.long)
        if unlabeled:
            idxos = get_value_list(item_list, 'idxo')  # [(N11o), (N12o), (N21o), (N22o), ..., (NB1o), (NB2o)], (2B)
        fns, fids = get_value_list(item_list, 'item_fn'), get_value_list(item_list, 'item_id')

        # voxelize
        voxel_locs, p2v_map, v2p_map = voxelization_idx(locs, self.batch_size * (1 if not unlabeled else 2), 4)
        # spatial shape
        spatial_shape = np.clip((locs.max(0)[0][1:] + 1).numpy(), self.full_scale[0], None)
        # batch offsets
        batch_offsets = [0]
        for value in get_value_list(item_list, 'xyz'):
            batch_offsets.append(batch_offsets[-1] + value.shape[0])
        batch_offsets = torch.tensor(batch_offsets, dtype=torch.int32)
        # idxos1, idxos2, point_cnts_o
        if unlabeled:
            idxos = [idxo + int(batch_offsets[i]) for i, idxo in enumerate(idxos)]
            idxos1 = cat_value_list(idxos[0::2], dtype=torch.long)
            idxos2 = cat_value_list(idxos[1::2], dtype=torch.long)
            point_cnts_o = torch.tensor([i.shape[0] for i in idxos][0::2], dtype=torch.int32)

        batch_data = {'locs': locs, 'voxel_locs': voxel_locs, 'p2v_map': p2v_map, 'v2p_map': v2p_map,
                      'locs_float': locs_float, 'feats': feats,
                      'offsets': batch_offsets, 'spatial_shape': spatial_shape,
                      'file_names': fns, 'id': fids}
        if not test:
            batch_data['labels'] = labels
        if unlabeled:
            batch_data['idxos'] = (idxos1, idxos2)
            batch_data['point_cnts_o'] = point_cnts_o

        return batch_data
