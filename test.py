import torch
import numpy as np
import random
import os
import torch.multiprocessing as mp
import torch.distributed as dist
import subprocess

import util.utils as utils
from model import build_network
from data import build_dataset
from util import commu_utils


def init_dist_pytorch(backend='nccl'):
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method('spawn')

    num_gpus = torch.cuda.device_count()
    assert cfg.batch_size % num_gpus == 0, f'Batch size should be matched with GPUS: ({cfg.batch_size}, {num_gpus})'
    cfg.batch_size = cfg.batch_size // num_gpus

    cfg.local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(cfg.local_rank)

    print(f'[PID {os.getpid()}] rank: {cfg.local_rank} world_size: {num_gpus}')
    dist.init_process_group(backend=backend)


def init_dist_slurm(backend='nccl'):
    proc_id = int(os.environ['SLURM_PROCID'])
    ntasks = int(os.environ['SLURM_NTASKS'])
    node_list = os.environ['SLURM_NODELIST']
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(proc_id % num_gpus)

    addr = subprocess.getoutput(f'scontrol show hostname {node_list} | head -n1')
    os.environ['MASTER_PORT'] = str(cfg.tcp_port)
    os.environ['MASTER_ADDR'] = addr
    os.environ['WORLD_SIZE'] = str(ntasks)
    os.environ['RANK'] = str(proc_id)
    dist.init_process_group(backend=backend)

    total_gpus = dist.get_world_size()
    assert cfg.batch_size % total_gpus == 0, f'Batch size should be matched with GPUS: ({cfg.batch_size}, {total_gpus})'
    cfg.batch_size = cfg.batch_size // total_gpus

    cfg.local_rank = dist.get_rank()


def init():
    cfg.task = 'test'
    cfg.result_dir = os.path.join(cfg.exp_path, 'result', f'iter{cfg.test_iter}', cfg.split)

    backup_dir = os.path.join(cfg.result_dir, 'backup_files')
    if cfg.local_rank == 0:
        os.makedirs(backup_dir, exist_ok=True)
        os.system(f'cp test.py {backup_dir}')
        os.system(f'cp {cfg.model_dir} {backup_dir}')
        os.system(f'cp {cfg.dataset_dir} {backup_dir}')
        os.system(f'cp {cfg.config} {backup_dir}')

    global logger
    from util.log import get_logger
    if cfg.local_rank == 0:
        logger = get_logger(cfg)
        logger.info(cfg)

    random.seed(cfg.manual_seed)
    np.random.seed(cfg.manual_seed)
    torch.manual_seed(cfg.manual_seed)
    torch.cuda.manual_seed_all(cfg.manual_seed)


def test(model, model_fn, dataset, dataloader, step):
    if cfg.local_rank == 0:
        logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')

    semantic_label_idx, semantic_names = utils.get_semantic_names(cfg.dataset)
    model.eval()
    semantic_scores_list = []
    with torch.no_grad():
        if cfg.dataset == 'semantic_kitti':
            augs = [[[1, 1], 0], [[1, 1], 0.1], [[1, 1], -0.1]]
        elif cfg.dataset in ['s3dis', 'scannetv2']:
            augs = [[1, 0], [1, 2 / 3], [1, 4 / 3]]
        else:
            raise NotImplementedError

        for rep in range(cfg.test_reps):
            intersection, union, target = [], [], []
            dataset.test_set.flipv, dataset.test_set.rotv = augs[rep][0], augs[rep][1]
            for batch_idx, batch in enumerate(dataloader):
                sample_ids, batch_offsets = batch['id'], batch['offsets']
                batch_size = len(sample_ids)
                N = batch['feats'].shape[0]

                # get predictions
                preds = model_fn(batch, model, step)
                semantic_scores = preds['semantic'].cpu()  # (N, nClass) float, cuda -> cpu
                if rep == 0:
                    semantic_scores_list.append(semantic_scores)
                else:
                    semantic_scores_list[batch_idx] += semantic_scores
                semantic_preds = semantic_scores_list[batch_idx].max(1)[1].numpy()  # (N) long, cpu -> numpy

                # prepare for evaluation
                if cfg.eval:
                    semantic_labels = batch['labels'].numpy()  # (N) long, numpy

                for b in range(batch_size):
                    cur_file_name = batch['file_names'][b]
                    cur_id = sample_ids[b]
                    cur_semantic_preds = semantic_preds[batch_offsets[b]: batch_offsets[b + 1]]

                    # prepare for evaluation
                    if cfg.eval:
                        cur_semantic_labels = semantic_labels[batch_offsets[b]: batch_offsets[b + 1]]
                        i, u, t = utils.intersectionAndUnion(cur_semantic_preds, cur_semantic_labels,
                                                             cfg.classes, cfg.ignore_label)
                        intersection.append(i)
                        union.append(u)
                        target.append(t)
                        acc = i.sum() / (t.sum() + 1e-10)
                        iou = (i[u != 0] / (u[u != 0] + 1e-10)).mean()
                        if cfg.local_rank == 0:
                            logger.info(
                                f'rep: {rep + 1}/{cfg.test_reps} iter: {batch_idx + 1}/{dataloader.__len__()} '
                                f'{cur_file_name}({cur_id}) acc: {acc:.4f} iou: {iou:.4f}')

                    # save files
                    if cfg.get('save_semantic', True):
                        if cfg.dataset == 'scannetv2':
                            cur_save_dir = os.path.join(cfg.result_dir, 'semantic', f'{cur_file_name[:12]}.npy')
                        elif cfg.dataset == 's3dis':
                            cur_save_dir = os.path.join(cfg.result_dir, 'semantic', f'{cur_file_name}.npy')
                        elif cfg.dataset == 'semantic_kitti':
                            sequence, frame = cur_file_name.split('_')
                            cur_save_dir = os.path.join(cfg.result_dir, 'semantic', sequence, f'{frame}.npy')

                        os.makedirs(os.path.dirname(cur_save_dir), exist_ok=True)
                        np.save(cur_save_dir, cur_semantic_preds)

                if cfg.local_rank == 0:
                    logger.info(f'rep: {rep + 1}/{cfg.test_reps} iter: {batch_idx + 1}/{dataloader.__len__()} point_num: {N}')

            if cfg.dist:
                intersection = commu_utils.merge_results_dist(intersection, len(dataset.test_file_names), cfg.result_dir)
                union = commu_utils.merge_results_dist(union, len(dataset.test_file_names), cfg.result_dir)
                target = commu_utils.merge_results_dist(target, len(dataset.test_file_names), cfg.result_dir)

            intersection = np.concatenate(intersection).reshape(-1, cfg.classes).sum(0)
            union = np.concatenate(union).reshape(-1, cfg.classes).sum(0)
            target = np.concatenate(target).reshape(-1, cfg.classes).sum(0)

            # evaluation
            if cfg.local_rank == 0:
                iou_class = intersection / (union + 1e-10)
                accuracy_class = intersection / (target + 1e-10)
                mIoU = np.mean(iou_class)
                mAcc = np.mean(accuracy_class)
                allAcc = sum(intersection) / (sum(target) + 1e-10)
                logger.info(f'mIoU/mAcc/allAcc {mIoU * 100:.2f}/{mAcc * 100:.2f}/{allAcc * 100:.2f}.')

        if cfg.local_rank == 0:
            utils.print_iou_acc_class(iou_class, accuracy_class, semantic_names, logger)
            logger.info(f'mIoU/mAcc/allAcc {mIoU * 100:.2f}/{mAcc * 100:.2f}/{allAcc * 100:.2f}.')
    

if __name__ == '__main__':
    # config
    global cfg
    from util.config import get_parser
    cfg = get_parser()

    # init
    if cfg.launcher == 'pytorch':
        init_dist_pytorch(backend='nccl')
        cfg.dist = True
    elif cfg.launcher == 'slurm':
        init_dist_slurm(backend='nccl')
        cfg.dist = True
    else:
        cfg.dist = False
    init()

    # get model and data version
    exp_name = cfg.config.split('/')[-1][:-5]
    cfg.model_name = exp_name.split('_')[0]
    cfg.data_name = exp_name.split('_')[-1]

    # model
    if cfg.local_rank == 0:
        logger.info('=> creating model ...')
        logger.info(f'Classes: {cfg.classes}')

    cfg.pretrain_path = None
    model, model_fn = build_network(cfg, test=True)

    use_cuda = torch.cuda.is_available()
    if cfg.local_rank == 0:
        logger.info(f'cuda available: {use_cuda}')
    assert use_cuda
    model = model.cuda()

    if cfg.dist:
        num_gpus = torch.cuda.device_count()
        local_rank = cfg.local_rank % num_gpus
        if cfg.sync_bn:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(local_rank)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)

    if cfg.local_rank == 0:
        logger.info(f'#model parameters: {sum([x.nelement() for x in model.parameters()])}')

    # load model
    _, f = utils.checkpoint_restore(model, cfg.exp_path, cfg.config.split('/')[-1][:-5], cfg.test_iter,
                                    f=cfg.pretrain, dist=cfg.dist, gpu=cfg.local_rank % torch.cuda.device_count())
    if cfg.local_rank == 0:
        logger.info(f'Restore from {f}')

    # data
    dataset = build_dataset(cfg, test=True)
    dataset.testLoader()
    dataloader = dataset.test_data_loader
    if cfg.local_rank == 0:
        logger.info(f'Testing samples ({cfg.split}): {dataset.test_file_names.__len__()}')

    # evaluate
    test(model, model_fn, dataset, dataloader, cfg.test_iter)
