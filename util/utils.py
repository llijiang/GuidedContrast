import torch, glob, os, numpy as np, SharedArray as SA, math
import sys
sys.path.append('../')

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * float(n)
        self.count += n
        self.avg = self.sum / float(self.count)


def intersectionAndUnion(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert (output.ndim in [1, 2, 3])
    assert output.shape == target.shape
    output = output.reshape(output.size).copy()
    target = target.reshape(target.size)
    output[np.where(target == ignore_index)[0]] = ignore_index
    intersection = output[np.where(output == target)[0]]
    area_intersection, _ = np.histogram(intersection, bins=np.arange(K+1))
    # area_intersection: K, indicates the number of members in each class in intersection
    area_output, _ = np.histogram(output, bins=np.arange(K+1))
    area_target, _ = np.histogram(target, bins=np.arange(K+1))
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target


def checkpoint_restore(model, exp_path, exp_name, it=0, f='', dist=False, gpu=0, optimizer=None):
    if not f:
        if it > 0:
            f = os.path.join(exp_path, f'{exp_name}-{it:09d}.pth')
            assert os.path.isfile(f)
        else:
            f_list = sorted(glob.glob(os.path.join(exp_path, f'{exp_name}-*.pth')))
            if len(f_list) > 0:
                f = f_list[-1]
                it = int(f[len(exp_path) + len(exp_name) + 2: -4])

    if len(f) > 0:
        map_location = {'cuda:0': f'cuda:{gpu}'} if gpu > 0 else None
        state = torch.load(f, map_location=map_location)
        checkpoint = state if not (isinstance(state, dict) and 'state_dict' in state) else state['state_dict']
        for k, v in checkpoint.items():
            if 'module.' in k:
                checkpoint = {k[len('module.'):]: v for k, v in checkpoint.items()}
            break
        model_dict = model.module.state_dict() if dist else model.state_dict()
        model_dict.update(checkpoint)
        if dist:
            model.module.load_state_dict(model_dict)
        else:
            model.load_state_dict(model_dict)

        if optimizer is not None:
            if isinstance(state, dict) and 'optimizer' in state:
                optimizer.load_state_dict(state['optimizer'])

    return it, f


def is_power2(num):
    return num != 0 and ((num & (num - 1)) == 0)


def is_multiple(num, multiple):
    return num > 0 and num % multiple == 0


def is_last(num, total_num, ratio=0.95):
    return num > int(total_num * ratio)


def checkpoint_save(model, optimizer, exp_path, exp_name, it, iters,
                    save_freq=50, keep_freq=1000, keep_last_ratio=0.95):
    f = None
    if is_multiple(it, save_freq):
        f = os.path.join(exp_path, f'{exp_name}-{it:09d}.pth')
        state = {
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(state, f)

        it = it - save_freq
        if not is_multiple(it, keep_freq) and not is_last(it, iters, keep_last_ratio):
            fd = os.path.join(exp_path, f'{exp_name}-{it:09d}.pth')
            if os.path.isfile(fd):
                os.remove(fd)

    return f


def load_model_param(model, pretrained_dict, prefix=''):
    # suppose every param in model should exist in pretrain_dict, but may differ in the prefix of the name
    # For example:    model_dict: '0.conv.weight'     pretrain_dict: 'FC_layer.0.conv.weight'
    model_dict = model.state_dict()
    len_prefix = 0 if len(prefix) == 0 else len(prefix) + 1
    pretrained_dict_filter = {k[len_prefix:]: v for k, v in pretrained_dict.items()
                              if k[len_prefix:] in model_dict and prefix in k}
    assert len(pretrained_dict_filter) > 0
    model_dict.update(pretrained_dict_filter)
    model.load_state_dict(model_dict)
    return len(pretrained_dict_filter), len(model_dict)


def write_obj(points, colors, out_filename):
    N = points.shape[0]
    fout = open(out_filename, 'w')
    for i in range(N):
        c = colors[i]
        fout.write('v %f %f %f %d %d %d\n' % (points[i, 0], points[i, 1], points[i, 2], c[0], c[1], c[2]))
    fout.close()


def sa_create(name, var):
    x = SA.create(name, var.shape, dtype=var.dtype)
    x[...] = var[...]
    x.flags.writeable = False
    return x


def create_shared_memory(file_names):
    for i, fname in enumerate(file_names):
        fn = fname.split('/')[-1].split('.')[0]
        if not os.path.exists(f'/dev/shm/{fn}_xyz'):
            print(f'[PID {os.getpid()}] {i} {fn}')
            data = torch.load(fname)
            sa_create(f'shm://{fn}_xyz', data[0])
            sa_create(f'shm://{fn}_rgb', data[1])
            if len(data) >= 3:
                sa_create(f'shm://{fn}_label', data[2])


def delete_shared_memory(file_names):
    for fname in file_names:
        fn = fname.split('/')[-1].split('.')[0]
        if os.path.exists(f'/dev/shm/{fn}_xyz'):
            SA.delete(f'shm://{fn}_xyz')
            SA.delete(f'shm://{fn}_rgb')
            if os.path.exists(f'/dev/shm/{fn}_label'):
                SA.delete(f'shm://{fn}_label')


def _calculate_fan_in_and_fan_out(tensor):
    dimensions = tensor.dim()
    if dimensions < 2:
        raise ValueError('Fan in and fan out can not be computed for tensor with fewer than 2 dimensions')

    if dimensions == 2:  # Linear
        fan_in = tensor.size(1)
        fan_out = tensor.size(0)
    else:
        num_input_fmaps = tensor.size(1)
        num_output_fmaps = tensor.size(2)
        receptive_field_size = tensor.size(0)
        fan_in = num_input_fmaps * receptive_field_size
        fan_out = num_output_fmaps * receptive_field_size

    return fan_in, fan_out


def _calculate_correct_fan(tensor, mode):
    mode = mode.lower()
    valid_modes = ['fan_in', 'fan_out']
    if mode not in valid_modes:
        raise ValueError(f'Mode {mode} not supported, please use one of {valid_modes}')

    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    return fan_in if mode == 'fan_in' else fan_out


def kaiming_normal_(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu'):
    fan = _calculate_correct_fan(tensor, mode)
    gain = torch.nn.init.calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    with torch.no_grad():
        return tensor.normal_(0, std)


def print_iou_acc_class(iou_class, acc_class, semantic_names, logger=None):
    sep = ''
    col1 = ':'
    lineLen = 64

    if logger is None:
        print('')
        print('#' * lineLen)
    else:
        logger.info('')
        logger.info('#' * lineLen)
    line = ''
    line += f"{'what':<15}{sep}{col1}"
    line += f"{'iou':>15}{sep}"
    line += f"{'acc':>15}{sep}"
    if logger is None:
        print(line)
        print('#' * lineLen)
    else:
        logger.info(line)
        logger.info('#' * lineLen)

    for (li, label_name) in enumerate(semantic_names):
        iou = iou_class[li]
        acc = acc_class[li]
        line = f'{label_name:<15}{sep}{col1}'
        line += f'{sep}{iou:>15.3f}{sep}'
        line += f'{sep}{acc:>15.3f}{sep}'
        if logger is None:
            print(line)
        else:
            logger.info(line)

    if logger is None:
        print('-' * lineLen)
    else:
        logger.info('-' * lineLen)
    line = f"{'average':<15}{sep}{col1}"
    line += f'{np.mean(iou_class):>15.3f}{sep}'
    line += f'{np.mean(acc_class):>15.3f}{sep}'
    if logger is None:
        print(line)
        print('')
    else:
        logger.info(line)
        logger.info('')


def get_semantic_names(dataset):
    if dataset == 'scannetv2':
        semantic_label_idx = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]
        semantic_names = np.array([
            'wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf', 'picture',
            'counter', 'desk', 'curtain', 'refridgerator', 'shower curtain', 'toilet', 'sink', 'bathtub',
            'otherfurniture'
        ])
    elif dataset == 's3dis':
        semantic_label_idx = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
        semantic_names = np.array([
            'ceiling', 'floor', 'wall', 'beam', 'column', 'window', 'door', 'chair', 'table', 'bookcase', 'sofa',
            'board', 'clutter'
        ])
    elif dataset == 'semantic_kitti':
        semantic_label_idx = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
        semantic_names = np.array([
            'car', 'bicycle', 'motorcycle', 'truck', 'other-vehicle', 'person', 'bicyclist', 'motorcyclist',
            'road', 'parking', 'sidewalk', 'other-ground', 'building', 'fence', 'vegetation', 'trunk',
            'terrain', 'pole', 'traffic-sign'
        ])
    else:
        raise NotImplementedError
    return semantic_label_idx, semantic_names