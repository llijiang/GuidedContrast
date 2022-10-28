def build_dataset(cfg, test=False):
    if cfg.data_name == 's3dis':
        from .s3dis import MyDataset as Dataset
    elif cfg.data_name == 'scannet':
        from .scannetv2 import MyDataset as Dataset
    elif cfg.data_name == 'semantickitti':
        from .semantic_kitti import MyDataset as Dataset
    else:
        raise NotImplementedError(f'{cfg.data_name} not implemented.')

    dataset = Dataset(cfg, test)

    return dataset