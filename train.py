import torch
import torch.optim as optim
import torch.multiprocessing as mp
import torch.distributed as dist
from tensorboardX import SummaryWriter
import numpy as np
import time, os, random, subprocess

import util.utils as utils
from model import build_network
from data import build_dataset
from util.lr import initialize_scheduler


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
    # backup
    backup_dir = os.path.join(cfg.exp_path, 'backup_files')
    if cfg.local_rank == 0:
        os.makedirs(backup_dir, exist_ok=True)
        os.system(f'cp train.py {backup_dir}')
        os.system(f'cp {cfg.model_dir} {backup_dir}')
        os.system(f'cp {cfg.dataset_dir} {backup_dir}')
        os.system(f'cp {cfg.config} {backup_dir}')

    if cfg.local_rank == 0:
        # logger
        global logger
        from util.log import get_logger
        logger = get_logger(cfg)
        logger.info(cfg)

        # summary writer
        global writer
        writer = SummaryWriter(cfg.exp_path)

    # random seed
    random.seed(cfg.manual_seed)
    np.random.seed(cfg.manual_seed)
    torch.manual_seed(cfg.manual_seed)
    torch.cuda.manual_seed_all(cfg.manual_seed)


def get_batch_data(dataloader, sampler, data_iterator=None, epoch=0, it_in_epoch=0, dist=False):
    if data_iterator is None or it_in_epoch == 0:
        if dist:
            sampler.set_epoch(epoch)
        data_iterator = iter(dataloader)

    batch = next(data_iterator)

    it_in_epoch = (it_in_epoch + 1) % (dataloader.__len__())
    epoch = epoch + int(it_in_epoch == 0)

    return batch, data_iterator, epoch, it_in_epoch


def evaluate(cfg, model, model_fn, dataloader, it):
    model.eval()

    if cfg.local_rank == 0:
        logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    am_dict = {}

    with torch.no_grad():
        start_epoch = time.time()
        for i, batch in enumerate(dataloader):
            # forward
            loss, preds, visual_dict, meter_dict = model_fn(batch, model, it)

            # merge multi-gpu
            if cfg.dist:
                for k, v in visual_dict.items():   # losses
                    count = meter_dict[k][1]

                    v = v * count
                    count = loss.new_tensor([count], dtype=torch.long)
                    dist.all_reduce(v), dist.all_reduce(count)
                    count = count.item()
                    v = v / count

                    visual_dict[k] = v
                    meter_dict[k] = (float(v), count)

            # meter_dict
            for k, v in meter_dict.items():
                if k not in am_dict.keys():
                    am_dict[k] = utils.AverageMeter()
                if cfg.dist and k in ['intersection', 'union', 'target']:
                    cnt_list = torch.from_numpy(v[0]).cuda()
                    dist.all_reduce(cnt_list)
                    am_dict[k].update(cnt_list.cpu().numpy(), v[1])
                else:
                    am_dict[k].update(v[0], v[1])

            # infos
            if cfg.local_rank == 0:
                print(f"{i + 1}/{dataloader.__len__()} loss: {am_dict['loss'].val:.4f}({am_dict['loss'].avg:.4f})")

        if cfg.local_rank == 0:
            print_info = f"iter: {it}/{cfg.iters}, val loss: {am_dict['loss'].avg:.4f}, " \
                         f"time: {time.time() - start_epoch:.4f}s"

            # summary writer
            for k in am_dict.keys():
                if k in visual_dict.keys():
                    writer.add_scalar(k + '_eval', am_dict[k].avg, it)

            if 'intersection' in am_dict:
                miou = (am_dict['intersection'].sum / (am_dict['union'].sum + 1e-10)).mean()
                macc = (am_dict['intersection'].sum / (am_dict['target'].sum + 1e-10)).mean()
                allacc = (am_dict['intersection'].sum).sum() / ((am_dict['target'].sum).sum() + 1e-10)
                writer.add_scalar('miou_eval', miou, it)
                writer.add_scalar('macc_eval', macc, it)
                writer.add_scalar('allacc_eval', allacc, it)
                print_info += f', miou: {miou:.4f}, macc: {macc:.4f}, allacc: {allacc:.4f}'

            logger.info(print_info)


def train_iter(cfg, model, model_fn, optimizer, scheduler, dataset, data_iterator_l, data_iterator_u,
               it, epoch_l, epoch_u, it_in_epoch_l, it_in_epoch_u,  # it from 1, epoch from 0, it_in_epoch from 0
               am_dict, train_with_unlabeled=False):
    end = time.time()

    # data
    batch_l, data_iterator_l, epoch_l, it_in_epoch_l = get_batch_data(
        dataset.l_train_data_loader, dataset.l_train_sampler, data_iterator_l, epoch_l, it_in_epoch_l, cfg.dist)
    batch = batch_l
    if train_with_unlabeled and it > cfg.prepare_iter:
        batch_u, data_iterator_u, epoch_u, it_in_epoch_u = get_batch_data(
            dataset.u_train_data_loader, dataset.u_train_sampler, data_iterator_u, epoch_u, it_in_epoch_u, cfg.dist)
        batch = (batch_l, batch_u)
    am_dict['data_time'].update(time.time() - end)

    # forward
    loss, _, visual_dict, meter_dict = model_fn(batch, model, it)

    # backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # adjust learning rate
    lrs = scheduler.get_last_lr()
    scheduler.step()

    # meter dict
    for k, v in meter_dict.items():
        if k not in am_dict.keys():
            am_dict[k] = utils.AverageMeter()
        am_dict[k].update(v[0], v[1])

    # summary writer
    if cfg.local_rank == 0:
        writer.add_scalar('lr_train', lrs[0], it)
        if epoch_l > 0 and it_in_epoch_l == 0:
            print_info = f'iter: {it}/{cfg.iters}'
            for k in am_dict.keys():
                if k in visual_dict.keys():
                    writer.add_scalar(k + '_train', am_dict[k].avg, it)
                    print_info += f', {k}: {am_dict[k].avg:.4f}'
            if 'intersection' in am_dict:
                miou = (am_dict['intersection'].sum / (am_dict['union'].sum + 1e-10)).mean()
                macc = (am_dict['intersection'].sum / (am_dict['target'].sum + 1e-10)).mean()
                allacc = (am_dict['intersection'].sum).sum() / ((am_dict['target'].sum).sum() + 1e-10)
                writer.add_scalar('miou_train', miou, it)
                writer.add_scalar('macc_train', macc, it)
                writer.add_scalar('allacc_train', allacc, it)
                print_info += f', miou: {miou:.4f}, macc: {macc:.4f}, allacc: {allacc:.4f}'
            logger.info(print_info)

    # save checkpoint
    if cfg.local_rank == 0:
        f = utils.checkpoint_save(model, optimizer, cfg.exp_path, cfg.config.split('/')[-1][:-5], it, cfg.iters,
                                  save_freq=cfg.save_freq, keep_freq=cfg.keep_freq, keep_last_ratio=cfg.keep_last_ratio)
        if f is not None:
            logger.info(f'iter: {it}/{cfg.iters}, Saving {f}')

    # infos
    am_dict['iter_time'].update(time.time() - end)

    remain_iter = cfg.iters - it
    remain_time = time.strftime('%d:%H:%M:%S', time.gmtime(remain_iter * am_dict['iter_time'].avg))
    remain_time = f'{int(remain_time[:2]) - 1:02d}{remain_time[2:]}'

    if cfg.local_rank == 0:
        logger.info(
            f"iter: {it}/{cfg.iters}, lr: {lrs[0]:.4e} "
            f"loss: {am_dict['loss'].val:.4f}({am_dict['loss'].avg:.4f}) "
            f"data_time: {am_dict['data_time'].val:.2f}({am_dict['data_time'].avg:.2f}) "
            f"iter_time: {am_dict['iter_time'].val:.2f}({am_dict['iter_time'].avg:.2f}) "
            f"remain_time: {remain_time}")

    # reset meter_dict
    if epoch_l > 0 and it_in_epoch_l == 0:
        for k in am_dict.keys():
            if k in visual_dict.keys():
                am_dict[k].reset()
        if 'intersection' in am_dict:
            am_dict['intersection'].reset(), am_dict['union'].reset(), am_dict['target'].reset()

    return data_iterator_l, data_iterator_u, epoch_l, epoch_u, it_in_epoch_l, it_in_epoch_u


def train(cfg, model, model_fn, optimizer, scheduler, dataset, start_iter=0):
    model.train()
    data_iterator_l, data_iterator_u = None, None
    epoch_l, it_in_epoch_l = divmod(start_iter, dataset.l_train_data_loader.__len__())
    epoch_u, it_in_epoch_u = divmod(max(start_iter - cfg.prepare_iter, 0), dataset.u_train_data_loader.__len__())

    am_dict = {}
    am_dict['iter_time'] = utils.AverageMeter()
    am_dict['data_time'] = utils.AverageMeter()

    for it in range(start_iter, cfg.iters):  # start from 0
        data_iterator_l, data_iterator_u, epoch_l, epoch_u, it_in_epoch_l, it_in_epoch_u = train_iter(
            cfg, model, model_fn, optimizer, scheduler, dataset, data_iterator_l, data_iterator_u,
            it + 1, epoch_l, epoch_u, it_in_epoch_l, it_in_epoch_u, am_dict, train_with_unlabeled=cfg.semi)

        if cfg.validation and (
                utils.is_multiple(it + 1, cfg.eval_freq) or
                (utils.is_last(it + 1, cfg.iters, cfg.eval_last_ratio) and utils.is_multiple(it + 1, cfg.save_freq))):
            evaluate(cfg, model, model_fn, dataset.val_data_loader, it + 1)
            model.train()


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

    # get model version and data version
    exp_name = cfg.config.split('/')[-1][:-5]
    cfg.model_name = exp_name.split('_')[0]
    cfg.data_name = exp_name.split('_')[-1]

    # model
    if cfg.local_rank == 0:
        logger.info('=> creating model ...')

    model, model_fn = build_network(cfg)

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

    # optimizer
    params = filter(lambda p: p.requires_grad, model.parameters())
    if cfg.optim == 'Adam':
        optimizer = optim.Adam(params, lr=cfg.lr)
    elif cfg.optim == 'SGD':
        optimizer = optim.SGD(params, lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    else:
        raise NotImplementedError

    # dataset
    dataset = build_dataset(cfg)

    if cfg.dataset == 's3dis':
        if cfg.local_rank == 0:
            logger.info(f'Training area: {cfg.train_area}')
            logger.info(f'Validation area: {cfg.test_area}')
    dataset.trainLoader()
    dataset.valLoader()
    if cfg.local_rank == 0:
        logger.info(f'Training samples: {dataset.train_file_names.__len__()}')
        logger.info(f'Validation samples: {dataset.val_file_names.__len__()}')

    # resume
    start_iter, f = utils.checkpoint_restore(model, cfg.exp_path, cfg.config.split('/')[-1][:-5], dist=cfg.dist,
                                             gpu=cfg.local_rank % torch.cuda.device_count(), optimizer=optimizer)
    if cfg.local_rank == 0:
        logger.info(f'Restore from {f}' if len(f) > 0 else f'Start from iteration {start_iter}')

    # lr_scheduler
    scheduler = initialize_scheduler(optimizer, cfg, last_step=start_iter - 1)

    train(cfg, model, model_fn, optimizer, scheduler, dataset, start_iter=start_iter)  # start_iter from 0

