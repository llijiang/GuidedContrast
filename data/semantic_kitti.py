import os, sys
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import yaml
sys.path.append('../')

from lib.ops import voxelization_idx
from util.pointdata_process import random_flip_dim, global_rotation, global_scaling, \
    get_overlap_xy_fan_crops, get_xy_fan_crop, get_sample_list


class SemanticKitti(Dataset):
    def __init__(self, cfg, file_names, aug_options, test=False, unlabeled=False):
        super(SemanticKitti, self).__init__()

        self.point_cloud_range = cfg.point_cloud_range
        self.voxel_size = cfg.voxel_size if len(cfg.voxel_size) == 3 else [cfg.voxel_size[0]] * 3

        self.file_names = file_names

        self.flip, self.rot, self.scale, self.fan_crop = aug_options
        self.flipv, self.rotv = [None, None], None

        self.unlabeled = unlabeled
        self.crop_angle_range = cfg.crop_angle_range
        self.crop_max_iters = cfg.crop_max_iters
        self.crop_return_ori = cfg.crop_return_ori

        self.test = test

        semantic_kitti_cfg_file = os.path.join(cfg.data_root, cfg.dataset, 'semantic-kitti.yaml')
        with open(semantic_kitti_cfg_file, 'r') as stream:
            semkittiyaml = yaml.safe_load(stream)
        self.learning_map = semkittiyaml['learning_map']

    def item_process(self, points, labels=None):
        if self.flip:
            points = random_flip_dim(points, dim=1, flipv=self.flipv[0])
            points = random_flip_dim(points, dim=0, flipv=self.flipv[1])
        if self.rot:
            points = global_rotation(points, rot_range=[-0.25, 0.25], rotv=self.rotv)   # -45 ~ 45
        if self.scale:
            points = global_scaling(points, scale_range=[0.95, 1.05])

        points[:, 0] = np.clip(points[:, 0], self.point_cloud_range[0] + 1e-3, self.point_cloud_range[3] - 1e-3)
        points[:, 1] = np.clip(points[:, 1], self.point_cloud_range[1] + 1e-3, self.point_cloud_range[4] - 1e-3)
        points[:, 2] = np.clip(points[:, 2], self.point_cloud_range[2] + 1e-3, self.point_cloud_range[5] - 1e-3)

        xyz_float = points[:, :3]
        feats = points[:, 3:4]
        xyz = (xyz_float - xyz_float.min(0)) / self.voxel_size

        item = {'xyz': xyz, 'xyz_float': xyz_float, 'feat': feats}
        if not self.test:
            item['label'] = labels

        return item

    def __getitem__(self, id):
        fn = self.file_names[id].split('/')[-3:]
        fn = '_'.join([fn[0], fn[-1].split('.')[0]])

        points = np.fromfile(self.file_names[id], dtype=np.float32).reshape((-1, 4))
        if not self.test:
            labels = np.fromfile(self.file_names[id].replace('velodyne', 'labels')[:-3] + 'label', dtype=np.int32)
            labels = labels & 0xFFFF  # delete high 16 digits binary
            labels = np.vectorize(self.learning_map.__getitem__)(labels)
            labels = labels - 1  # shift [0, 19] to [-1, 18]

        if self.unlabeled:
            idx1, idx2, idxo, idxo1, idxo2, points1, points2 = get_overlap_xy_fan_crops(
                points, self.crop_angle_range, self.crop_max_iters, self.crop_return_ori)

            if not self.test:
                labels1 = labels[idx1].copy()
                labels2 = labels[idx2].copy()

            item1 = self.item_process(points1, labels=None if self.test else labels1)
            item2 = self.item_process(points2, labels=None if self.test else labels2)

            item = {'idx': (idx1, idx2), 'idxo': (idxo1, idxo2)}
            for k in item1.keys():
                item[k] = (item1[k], item2[k])

        else:
            if self.fan_crop:
                idx, points = get_xy_fan_crop(points, self.crop_angle_range, self.crop_return_ori)

                if not self.test:
                    labels = labels[idx]

            item = self.item_process(points, labels=None if self.test else labels)

        item.update({'item_id': id, 'item_fn': fn})
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

        self.train_flip, self.train_rot, self.train_scale = cfg.train_flip, cfg.train_rot, cfg.train_scale
        self.train_fan_crop = cfg.train_fan_crop

        self.point_cloud_range = cfg.point_cloud_range
        self.voxel_size = cfg.voxel_size if len(cfg.voxel_size) == 3 else [cfg.voxel_size[0]] * 3

        self.labeled_ratio = cfg.labeled_ratio

        if test:
            self.test_split = cfg.split  # valid or test
            self.test_workers = cfg.test_workers

        semantic_kitti_cfg_file = os.path.join(self.data_root, self.dataset, 'semantic-kitti.yaml')
        with open(semantic_kitti_cfg_file, 'r') as stream:
            self.semkittiyaml = yaml.safe_load(stream)

    def trainLoader(self):
        self.train_file_names = get_sample_list(
            data_dir=os.path.join(self.data_root, self.dataset),
            split=self.semkittiyaml['split']['train']
        )

        if self.labeled_ratio < 1:
            split_path = os.path.join(self.data_root, self.dataset, 'data_split')
            split_fn = os.path.join(split_path, f'{int(self.labeled_ratio * 100)}.txt')
            with open(split_fn) as f:
                self.labeled_fns = f.readlines()
                self.labeled_fns = [i.strip() for i in self.labeled_fns]
            self.unlabeled_fns = [fn for fn in self.train_file_names if fn not in self.labeled_fns]
        else:
            self.labeled_fns, self.unlabeled_fns = self.train_file_names, self.train_file_names

        l_train_set = SemanticKitti(
            self.cfg, self.labeled_fns,
            [self.train_flip, self.train_rot, self.train_scale, self.train_fan_crop],
            unlabeled=False
        )

        u_train_set = SemanticKitti(
            self.cfg, self.unlabeled_fns,
            [self.train_flip, self.train_rot, self.train_scale, self.train_fan_crop],
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
        self.val_file_names = get_sample_list(
            data_dir=os.path.join(self.data_root, self.dataset),
            split=self.semkittiyaml['split']['valid']
        )
        val_set = SemanticKitti(
            self.cfg, self.val_file_names, [False, False, False, False]
        )
        self.val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_set, shuffle=False, drop_last=False) if self.dist else None
        self.val_data_loader = DataLoader(val_set, batch_size=self.batch_size, collate_fn=self.get_batch_data,
                                          num_workers=self.val_workers, shuffle=False, sampler=self.val_sampler,
                                          drop_last=False, pin_memory=True, worker_init_fn=self._worker_init_fn_)

    def testLoader(self):
        assert self.test_split in ['valid', 'test']
        self.test_file_names = get_sample_list(
            data_dir=os.path.join(self.data_root, self.dataset),
            split=self.semkittiyaml['split'][self.test_split]
        )
        self.test_set = SemanticKitti(
            self.cfg, self.test_file_names, [True, True, False, False], test=(self.test_split == 'test')
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
        feats = cat_value_list(get_value_list(item_list, 'feat'))
        if not test:
            labels = cat_value_list(get_value_list(item_list, 'label'), dtype=torch.long)
        if unlabeled:
            idxos = get_value_list(item_list, 'idxo')  # [(N11o), (N12o), (N21o), (N22o), ..., (NB1o), (NB2o)], (2B)
        fns, fids = get_value_list(item_list, 'item_fn'), get_value_list(item_list, 'item_id')

        # voxelize
        voxel_locs, p2v_map, v2p_map = voxelization_idx(locs, self.batch_size * (1 if not unlabeled else 2), 4)
        # spatial shape
        point_cloud_range = np.array(self.point_cloud_range)
        spatial_shape = ((point_cloud_range[3:6] - point_cloud_range[0:3]) / np.array(self.voxel_size)).astype(np.int)
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
