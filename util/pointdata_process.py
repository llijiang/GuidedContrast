import scipy.ndimage
import scipy.interpolate
import numpy as np
import torch
import torch.distributed as dist
import glob, random, math
import sys, os
sys.path.append('../')

from util import utils
from util import commu_utils

'''for indoor datasets'''
def dataAugment(xyz, jitter=False, flip=False, rot=False, flipv=None, rotv=None):
    m = np.eye(3)
    if jitter:
        m += np.random.randn(3, 3) * 0.1
    if flip:    # flip x
        if flipv is None:
            flipv = np.random.randint(0, 2) * 2 - 1   # 1 / -1
        m[0][0] *= flipv
    if rot:
        if rotv is None:
            rotv = np.random.rand() * 2    # 0 ~ 2
        rotv = rotv * math.pi
        m = np.matmul(m, [[math.cos(rotv), math.sin(rotv), 0],
                          [-math.sin(rotv), math.cos(rotv), 0],
                          [0, 0, 1]])
    return np.matmul(xyz, m)


def elastic(x, gran, mag):
    blur0 = np.ones((3, 1, 1)).astype('float32') / 3
    blur1 = np.ones((1, 3, 1)).astype('float32') / 3
    blur2 = np.ones((1, 1, 3)).astype('float32') / 3

    bb = (np.abs(x).max(0).astype(np.int32) // gran + 3).astype(np.int32)
    noise = [np.random.randn(bb[0], bb[1], bb[2]).astype('float32') for _ in range(3)]
    noise = [scipy.ndimage.filters.convolve(n, blur0, mode='constant', cval=0) for n in noise]
    noise = [scipy.ndimage.filters.convolve(n, blur1, mode='constant', cval=0) for n in noise]
    noise = [scipy.ndimage.filters.convolve(n, blur2, mode='constant', cval=0) for n in noise]
    noise = [scipy.ndimage.filters.convolve(n, blur0, mode='constant', cval=0) for n in noise]
    noise = [scipy.ndimage.filters.convolve(n, blur1, mode='constant', cval=0) for n in noise]
    noise = [scipy.ndimage.filters.convolve(n, blur2, mode='constant', cval=0) for n in noise]
    ax = [np.linspace(-(b - 1) * gran, (b - 1) * gran, b) for b in bb]
    interp = [scipy.interpolate.RegularGridInterpolator(ax, n, bounds_error=0, fill_value=0) for n in noise]

    def g(x_):
        return np.hstack([i(x_)[:, None] for i in interp])

    return x + g(x) * mag


def get_xy_crop(xyz, crop_size):
    crop_size = np.array(crop_size)
    half_cs = crop_size / 2.0
    xy_center_min = xyz.min(0)[:2] + half_cs - 0.01
    xy_center_max = xyz.max(0)[:2] - half_cs + 0.01

    selected_idx = random.randint(0, xyz.shape[0] - 1)
    selected_pt = xyz[selected_idx]

    center = selected_pt[:2].copy()
    center = np.clip(center, xy_center_min, xy_center_max)

    mask = (xyz[:, 0] >= center[0] - half_cs[0] - 0.001) * (xyz[:, 0] <= center[0] + half_cs[0] + 0.001) * \
           (xyz[:, 1] >= center[1] - half_cs[1] - 0.001) * (xyz[:, 1] <= center[1] + half_cs[1] + 0.001)

    idx = np.where(mask)[0]

    return idx


def get_overlap_xy_crops(xyz, crop_size, max_iters=50):
    crop_size = np.array(crop_size)
    half_cs = crop_size / 2.0
    xy_center_min = xyz.min(0)[:2] + half_cs - 0.01
    xy_center_max = xyz.max(0)[:2] - half_cs + 0.01

    selected_idx = random.randint(0, xyz.shape[0] - 1)
    selected_pt = xyz[selected_idx]

    center1 = selected_pt[:2].copy()
    center1[0] = random.uniform(selected_pt[0] - half_cs[0] + 0.01, selected_pt[0] + half_cs[0] - 0.01)
    center1[1] = random.uniform(selected_pt[1] - half_cs[1] + 0.01, selected_pt[1] + half_cs[1] - 0.01)
    center1 = np.clip(center1, xy_center_min, xy_center_max)

    t = 0
    while t < max_iters:
        center2 = selected_pt[:2].copy()
        center2[0] = random.uniform(selected_pt[0] - half_cs[0] + 0.01, selected_pt[0] + half_cs[0] - 0.01)
        center2[1] = random.uniform(selected_pt[1] - half_cs[1] + 0.01, selected_pt[1] + half_cs[1] - 0.01)
        center2 = np.clip(center2, xy_center_min, xy_center_max)

        inter = max(0, (crop_size[0] - abs(center1[0] - center2[0]))) * \
                max(0, (crop_size[1] - abs(center1[1] - center2[1])))
        union = 2 * crop_size[0] * crop_size[1] - inter
        iou = inter / union
        if iou >= 0.1 and iou <= 0.99:
            break

        t += 1

    if t == max_iters:
        center2 = center1

    mask1 = (xyz[:, 0] >= center1[0] - half_cs[0] - 0.001) * (xyz[:, 0] <= center1[0] + half_cs[0] + 0.001) * \
            (xyz[:, 1] >= center1[1] - half_cs[1] - 0.001) * (xyz[:, 1] <= center1[1] + half_cs[1] + 0.001)
    mask2 = (xyz[:, 0] >= center2[0] - half_cs[0] - 0.001) * (xyz[:, 0] <= center2[0] + half_cs[0] + 0.001) * \
            (xyz[:, 1] >= center2[1] - half_cs[1] - 0.001) * (xyz[:, 1] <= center2[1] + half_cs[1] + 0.001)
    masko = mask1 & mask2
    masko1 = masko[mask1]
    masko2 = masko[mask2]

    assert masko.sum() > 0, f'no overlapped points: mask1 {mask1.sum()} mask2 {mask2.sum()} masko {masko.sum()}'
    assert masko1.sum() == masko.sum()
    assert masko2.sum() == masko.sum()

    idx1 = np.where(mask1)[0]     # idx in N
    idx2 = np.where(mask2)[0]     # idx in N
    idxo = np.where(masko)[0]     # idx in N
    idxo1 = np.where(masko1)[0]     # idx in N1
    idxo2 = np.where(masko2)[0]     # idx in N2

    return idx1, idx2, idxo, idxo1, idxo2


'''for unsup loss'''
def split_embed(embed, offsets):
    '''
    :param embed: (N11 + N12 + N21 + N22 + ... + NB1 + NB2, C)
    :param offsets: (2B + 1)
    :return embed1: (N11 + N21 + ... + NB1, C)
    :return embed2: (N12 + N22 + ... + NB2, C)
    '''
    batch_size = len(offsets) - 1
    assert batch_size % 2 == 0
    batch_size = int(batch_size / 2)

    embed1 = []
    embed2 = []
    for i in range(batch_size):
        embed1.append(embed[offsets[2 * i]: offsets[2 * i + 1]])
        embed2.append(embed[offsets[2 * i + 1]: offsets[2 * i + 2]])
    embed1 = torch.cat(embed1, 0)
    embed2 = torch.cat(embed2, 0)
    return embed1, embed2


def get_pos_sample_idx(cnts, num_samples, labels, num_classes):
    '''
    Category-balanced Sampling
    :param cnts: [N1, ..., Nn], int, cuda
    :param num_samples: M
    :param labels: (N1 + ... + Nn), long, cuda
    :return sample_idx: (n * M), long, cuda
    '''
    cnts_samples = torch.tensor([num_samples] * cnts.__len__(), dtype=torch.int32, device='cuda')

    sample_idx = []

    offset = 0
    for cnt, cnt_samples in zip(cnts, cnts_samples):
        cnt = int(cnt)
        cnt_samples = int(cnt_samples)
        cnt_samples_per_class = cnt_samples // num_classes
        cur_labels = labels[offset: offset + cnt]

        num_samples = 0
        for k in range(num_classes):
            sidx = torch.where(cur_labels == k)[0]
            snum = sidx.__len__()
            if snum >= cnt_samples_per_class:
                perm = torch.randperm(snum)[:cnt_samples_per_class]
                sidx = sidx[perm]

            sample_idx.append(sidx + offset)
            num_samples += sidx.__len__()

        num_res = cnt_samples - num_samples
        sidx = torch.randperm(cnt, device='cuda')[:num_res]
        sample_idx.append(sidx + offset)

        offset += cnt

    sample_idx = torch.cat(sample_idx)
    return sample_idx


def get_neg_sample(embed_u, indices, pseudo_labels, embed_bank, index_bank, queue_ptr, valid_ptr, cfg):
    '''
    :param embed_u: (Nu, C), float, cuda
    :param indices: (Nu), long, cuda
    :param pseudo_labels: (Nu), long, cuda
    :param embed_bank: (num_classes, bank_length, C), float, cuda
    :param index_bank: (num_classes, bank_length), long, cuda
    :param queue_ptr: (num_classes), long, cuda
    :param valid_ptr: (num_classes), long, cuda
    :return embed_neg
    :return pseudo_labels_neg
    :return neg_idx
    '''
    bank_enqueue(embed_u.detach(), indices, pseudo_labels, embed_bank, index_bank, queue_ptr, valid_ptr,
                 num_classes=cfg.classes, max_num_enqueue_per_class=cfg.max_num_enqueue_per_class)
    embed_neg, neg_idx, pseudo_labels_neg = bank_fetch(embed_bank, index_bank, queue_ptr, valid_ptr, cfg.classes,
        num_selected=cfg.num_neg_sample // (1 if not cfg.dist else dist.get_world_size()))

    if cfg.dist:
        embed_neg = commu_utils.concat_all_gather(embed_neg)   # (num_gpu * num_selected, C), float, cuda
        pseudo_labels_neg = commu_utils.concat_all_gather(pseudo_labels_neg)   # (num_gpu * num_selected), long, cuda
        neg_idx_ = torch.ones(embed_neg.shape[0], dtype=neg_idx.dtype, device=neg_idx.device) * (-1)
        neg_idx_[neg_idx.shape[0] * cfg.local_rank: neg_idx.shape[0] * (cfg.local_rank + 1)] = neg_idx
        neg_idx = neg_idx_   # (num_gpu * num_selected), long, cuda, idx in Nu
    embed_neg = embed_neg.detach()

    return embed_neg, pseudo_labels_neg, neg_idx


def bank_enqueue(embed, indices, labels, embed_bank, index_bank, queue_ptr, valid_ptr,
                 num_classes, max_num_enqueue_per_class=100):
    '''
    :param embed: (N, C), float, cuda
    :param indices: (N), long, cuda
    :param labels: (N), long, cuda
    :param embed_bank: (num_classes, bank_length, C), float, cuda
    :param index_bank: (num_classes, bank_length), long, cuda
    :param queue_ptr: (num_classes), long, cuda
    :param valid_ptr: (num_classes), long, cuda
    :param num_classes:
    :param max_num_enqueue_per_class:
    '''
    bank_length = embed_bank.shape[1]

    for k in range(num_classes):
        src_idx = (labels == k).nonzero().view(-1)
        cur_num_enqueue = min(src_idx.__len__(), max_num_enqueue_per_class)
        perm = torch.randperm(src_idx.__len__())[:cur_num_enqueue]
        sample_idx = src_idx[perm]
        cur_embed = embed[sample_idx]
        cur_indices = indices[sample_idx]

        ptr = int(queue_ptr[k])
        if ptr + cur_num_enqueue <= bank_length:
            embed_bank[k, ptr: ptr + cur_num_enqueue] = cur_embed
            index_bank[k, ptr: ptr + cur_num_enqueue] = cur_indices
            valid_ptr[k] = max(int(valid_ptr[k]), ptr + cur_num_enqueue)
        else:
            embed_bank[k, ptr:] = cur_embed[:bank_length - ptr]
            embed_bank[k, 0: cur_num_enqueue - (bank_length - ptr)] = cur_embed[bank_length - ptr:]
            index_bank[k, ptr:] = cur_indices[:bank_length - ptr]
            index_bank[k, 0: cur_num_enqueue - (bank_length - ptr)] = cur_indices[bank_length - ptr:]
            valid_ptr[k] = bank_length
        queue_ptr[k] = (ptr + cur_num_enqueue) % bank_length


def bank_fetch(embed_bank, index_bank, queue_ptr, valid_ptr, num_classes, num_selected):
    '''
    :param embed_bank: (num_classes, bank_length, C), float, cuda
    :param index_bank: (num_classes, bank_length), long, cuda
    :param queue_ptr: (num_classes), long, cuda
    :param valid_ptr: (num_classes), long, cuda
    :return selected_embed: (num_selected, C), float, cuda
    :return selected_indices: (num_selected), long, cuda
    :return selected_labels: (num_selected), long, cuda
    '''
    selected_embed = []
    selected_indices = []
    selected_labels = []
    num_selected_per_class = num_selected // num_classes

    for k in range(num_classes):
        num_valid = max(int(valid_ptr[k]), num_selected_per_class)
        perm = torch.randperm(num_valid)[:num_selected_per_class]
        selected_embed.append(embed_bank[k][perm])
        selected_indices.append(index_bank[k][perm])
        selected_labels.append(torch.ones(num_selected_per_class, dtype=torch.int64, device='cuda') * k)

    selected_embed = torch.cat(selected_embed, dim=0)
    selected_indices = torch.cat(selected_indices, dim=0)
    selected_labels = torch.cat(selected_labels, dim=0)

    return selected_embed, selected_indices, selected_labels


'''for SemanticKitti'''
def random_flip_dim(points, dim=0, flipv=None):
    '''
    :param points: (N, 3 + C)
    '''
    if flipv is None:
        flipv = np.random.randint(0, 2) * 2 - 1  # 1 / -1
    points[:, dim] = points[:, dim] * flipv

    return points


def global_rotation(points, rot_range, rotv=None):
    '''
    :param points: (N, 3 + C)
    '''
    if rotv is None:
        rotv = np.random.uniform(rot_range[0], rot_range[1])   # -0.25 ~ 0.25
    rotv = rotv * math.pi  # -45 ~ 45
    points = rotate_points_along_z(points[np.newaxis, :, :], np.array([rotv]))[0]
    return points


def rotate_points_along_z(points, angle):
    '''
    :param points: (B, N, 3 + C)
    :param angle: (B), angle along z-axis, angle increases x -> y
    '''
    points, is_numpy = check_numpy_to_torch(points)
    angle, _ = check_numpy_to_torch(angle)

    cosa = torch.cos(angle)
    sina = torch.sin(angle)
    zeros = angle.new_zeros(points.shape[0])
    ones = angle.new_ones(points.shape[0])
    rot_matrix = torch.stack((
        cosa,  sina, zeros,
        -sina, cosa, zeros,
        zeros, zeros, ones
    ), dim=1).view(-1, 3, 3).float()
    points_rot = torch.matmul(points[:, :, 0:3], rot_matrix)
    points_rot = torch.cat((points_rot, points[:, :, 3:]), dim=-1)
    return points_rot.numpy() if is_numpy else points_rot


def check_numpy_to_torch(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float(), True
    return x, False


def global_scaling(points, scale_range):
    '''
    :param points: (N, 3 + C)
    '''
    if scale_range[1] - scale_range[0] < 1e-3:
        return points
    noise_scale = np.random.uniform(scale_range[0], scale_range[1])
    points[:, :3] *= noise_scale
    return points


def extract_fov_data(points, fov_degree, heading_angle, return_ori=False):
    '''
    :param points: (N, 3 + C)
    :param fov_degree: 0 ~ 360
    :param heading_angle: 0 ~ 360 in lidar coords, 0 is the x-axis, increase clockwise
    :param return_ori: if False, return rotated point cloud centered in heading angle
    '''

    def rotate_pc_along_z(pc, rot_angle):
        '''
        :param pc: (N, 3 + C), (N, 3) is in the LiDAR coordinate
        :param rot_angle: rad scalar
        :return pc: updated pc with XYZ rotated
        '''
        cosval = np.cos(rot_angle)
        sinval = np.sin(rot_angle)
        rotmat = np.array([[cosval, -sinval], [sinval, cosval]])
        pc[:, 0:2] = np.dot(pc[:, 0:2], rotmat)
        return pc

    half_fov_degree = fov_degree / 180 * np.pi / 2
    heading_angle = -heading_angle / 180 * np.pi
    points_new = rotate_pc_along_z(points.copy(), heading_angle)
    angle = np.arctan2(points_new[:, 1], points_new[:, 0])
    fov_mask = ((np.abs(angle) < half_fov_degree))
    if return_ori:
        points_ret = points[fov_mask]
    else:
        points_ret = points_new[fov_mask]
    return points_ret, fov_mask


def get_rad_union(l1, r1, l2, r2):
    '''
    :param l1, r1, l2, r2: [0, 360)
    '''
    if l1 < r1 and l2 < r2:
        union = max(0, min(r1, r2) - max(l1, l2))
    elif l1 >= r1 and l2 < r2:
        l10, r10 = l1, 360.0
        l11, r11 = 0, r1
        union = max(0, min(r10, r2) - max(l10, l2)) + max(0, min(r11, r2) - max(l11, l2))
    elif l1 < r1 and l2 >= r2:
        l20, r20 = l2, 360.0
        l21, r21 = 0, r2
        union = max(0, min(r1, r20) - max(l1, l20)) + max(0, min(r1, r21) - max(l1, l21))
    else:
        l10, r10 = l1, 360.0
        l11, r11 = 0, r1
        l20, r20 = l2, 360.0
        l21, r21 = 0, r2
        union = max(0, min(r10, r20) - max(l10, l20)) + max(0, min(r11, r20) - max(l11, l20)) + \
                max(0, min(r10, r21) - max(l10, l21)) + max(0, min(r11, r21) - max(l11, l21))
    return union


def get_overlap_xy_fan_crops(points, crop_angle_range, max_iters=50, return_ori=True):
    '''
    :param points: (N, 3 + C)
    :param crop_angle_range: (2), e.g., [120, 360]
    :param max_iters:
    :return:
    '''

    selected_idx = random.randint(0, points.shape[0] - 1)
    selected_pt = points[selected_idx]

    angle = np.arctan2(selected_pt[1], selected_pt[0])   # -pi ~ pi
    angle = (angle + 2 * np.pi) % (2 * np.pi) / np.pi * 180   # 0 ~ 360

    fov_degree1 = np.random.uniform(crop_angle_range[0], crop_angle_range[1])
    fov_degree2 = np.random.uniform(crop_angle_range[0], crop_angle_range[1])

    heading_angle1 = np.random.uniform(angle - fov_degree1 / 2.0 + 0.01, angle + fov_degree1 / 2.0 - 0.01) % 360.0
    l1 = (heading_angle1 - fov_degree1 / 2.0) % 360.0
    r1 = (heading_angle1 + fov_degree1 / 2.0) % 360.0

    t = 0
    while t < max_iters:
        heading_angle2 = np.random.uniform(angle - fov_degree2 / 2.0 + 0.01, angle + fov_degree2 / 2.0 - 0.01) % 360.0
        l2 = (heading_angle2 - fov_degree2 / 2.0) % 360.0
        r2 = (heading_angle2 + fov_degree2 / 2.0) % 360.0

        inter = get_rad_union(l1, r1, l2, r2)
        union = fov_degree1 + fov_degree2 - inter
        iou = inter / union
        if iou >= 0.1 and iou <= 0.99:
            break

        t += 1

    if t == max_iters:
        heading_angle2 = heading_angle1

    points1, mask1 = extract_fov_data(points, fov_degree1, heading_angle1, return_ori=return_ori)
    points2, mask2 = extract_fov_data(points, fov_degree2, heading_angle2, return_ori=return_ori)

    masko = mask1 & mask2
    masko1 = masko[mask1]
    masko2 = masko[mask2]

    assert masko.sum() > 0, f'no overlapped points: mask1 {mask1.sum()} mask2 {mask2.sum()} masko {masko.sum()}'
    assert masko1.sum() == masko.sum()
    assert masko2.sum() == masko.sum()

    idx1 = np.where(mask1)[0]
    idx2 = np.where(mask2)[0]
    idxo = np.where(masko)[0]
    idxo1 = np.where(masko1)[0]
    idxo2 = np.where(masko2)[0]

    return idx1, idx2, idxo, idxo1, idxo2, points1, points2


def get_xy_fan_crop(points, crop_angle_range, return_ori=True):
    selected_idx = random.randint(0, points.shape[0] - 1)
    selected_pt = points[selected_idx]

    angle = np.arctan2(selected_pt[1], selected_pt[0])  # -pi ~ pi
    angle = (angle + 2 * np.pi) % (2 * np.pi) / np.pi * 180  # 0 ~ 360

    fov_degree = np.random.uniform(crop_angle_range[0], crop_angle_range[1])

    points, mask = extract_fov_data(points, fov_degree, angle, return_ori)
    idx = np.where(mask)[0]

    return idx, points


def get_sample_list(data_dir, split):
    sample_list = []
    for i_folder in split:
        cur_dir = os.path.join(data_dir, 'sequences', '%02d' % i_folder, 'velodyne', '*.bin')
        cur_files = sorted(glob.glob(cur_dir))
        sample_list += cur_files

    return sample_list
