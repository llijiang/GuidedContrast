import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import functools
import sys, os
sys.path.append('../../')

from util.spconv_utils import spconv
from lib.ops import voxelization
from util import utils
from model.unet.unet import UNet, ResidualBlock, VGGBlock
from util.pointdata_process import get_pos_sample_idx, get_neg_sample, split_embed


class SemSeg(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        input_c = cfg.input_channel + 3
        classes = cfg.classes

        m = cfg.m
        embed_m = cfg.embed_m

        self.pretrain_path = cfg.pretrain_path
        self.pretrain_module = cfg.pretrain_module

        # backbone
        nPlanes = [m, 2 * m, 3 * m, 4 * m, 5 * m, 6 * m, 7 * m]
        block_reps = cfg.block_reps
        block = ResidualBlock if cfg.block_residual else VGGBlock
        norm_fn = functools.partial(nn.BatchNorm1d, eps=1e-4, momentum=0.1)
        self.backbone = UNet(input_c, m, nPlanes, block_reps, block, norm_fn, cfg)

        # projector
        self.projector = nn.Linear(m, embed_m)

        # classifier
        self.classifier = nn.Linear(m, classes)

        # memory bank
        self.register_buffer('embed_queue1', torch.randn(classes, cfg.bank_length, embed_m))
        self.embed_queue1 = F.normalize(self.embed_queue1, p=2, dim=-1)
        self.register_buffer('index_queue1', torch.zeros((classes, cfg.bank_length), dtype=torch.long))
        self.register_buffer('queue_pointer1', torch.zeros(classes, dtype=torch.long))
        self.register_buffer('valid_pointer1', torch.zeros(classes, dtype=torch.long))

        self.register_buffer('embed_queue2', torch.randn(classes, cfg.bank_length, embed_m))
        self.embed_queue2 = F.normalize(self.embed_queue2, p=2, dim=-1)
        self.register_buffer('index_queue2', torch.zeros((classes, cfg.bank_length), dtype=torch.long))
        self.register_buffer('queue_pointer2', torch.zeros(classes, dtype=torch.long))
        self.register_buffer('valid_pointer2', torch.zeros(classes, dtype=torch.long))

        self.apply(self.set_bn_init)

        # load pretrain weights
        module_map = {'backbone': self.backbone, 'classifier': self.classifier, 'projector': self.projector}
        if self.pretrain_path is not None:
            map_location = {'cuda:0': f'cuda:{cfg.local_rank % torch.cuda.device_count()}'} if cfg.local_rank > 0 else None
            state = torch.load(self.pretrain_path, map_location=map_location)
            pretrain_dict = state if not 'state_dict' in state else state['state_dict']
            if 'module.' in list(pretrain_dict.keys())[0]:
                pretrain_dict = {k[len('module.'):]: v for k, v in pretrain_dict.items()}
            for m in self.pretrain_module:
                n1, n2 = utils.load_model_param(module_map[m], pretrain_dict, prefix=m)
                if cfg.local_rank == 0:
                    print(f'[PID {os.getpid()}] Load pretrained {m}: {n1}/{n2}')

    @staticmethod
    def set_bn_init(m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.0)
            m.bias.data.fill_(0.0)

    def encoder(self, batch):
        voxel_coords = batch['voxel_locs'].cuda()  # (M, 1 + 3), long, cuda
        p2v_map = batch['p2v_map'].cuda()          # (N), int, cuda
        v2p_map = batch['v2p_map'].cuda()          # (M, 1 + maxActive), int, cuda

        coords_float = batch['locs_float'].cuda()  # (N, 3), float, cuda
        feats = batch['feats'].cuda()              # (N, C), float, cuda
        feats = torch.cat((feats, coords_float), 1)
        voxel_feats = voxelization(feats, v2p_map, 4)  # (M, C), float, cuda

        spatial_shape = batch['spatial_shape']

        batch_size = len(batch['offsets']) - 1

        inp = spconv.SparseConvTensor(voxel_feats, voxel_coords.int(), spatial_shape, batch_size)
        out = self.backbone(inp)
        out = out.features[p2v_map.long()]

        return out

    def forward(self, batch_l, batch_u=None):
        ret= {}

        output_feats_l = self.encoder(batch_l)                # (Nl, C)
        semantic_scores_l = self.classifier(output_feats_l)   # (Nl, nClass)
        ret['semantic_scores_l'] = semantic_scores_l
        ret['semantic_features_l'] = output_feats_l

        if batch_u is not None:
            output_feats_u = self.encoder(batch_u)              # (Nu, C)
            semantic_scores_u = self.classifier(output_feats_u) # (Nu, nClass)
            ret['semantic_scores_u'] = semantic_scores_u
            ret['semantic_features_u'] = output_feats_u

            output_feats_u = self.projector(output_feats_u)        # (Nu, C), Nu = Nu11 + Nu12 + ... + NuB1 + NuB2
            embed_u = F.normalize(output_feats_u, p=2, dim=1)
            ret['embed_u'] = embed_u

        return ret


def model_fn_decorator(cfg, test=False):
    criterion = nn.CrossEntropyLoss(
        ignore_index=cfg.ignore_label,
        weight=None if not 'loss_weights_classes' in cfg else torch.tensor(cfg.loss_weights_classes).float()
    ).cuda()

    def model_fn(batch, model, step):
        # forward
        if isinstance(batch, tuple):
            batch_l, batch_u = batch
        else:
            batch_l, batch_u = batch, None

        ret = model(batch_l, batch_u)

        semantic_scores_l = ret['semantic_scores_l']  # (Nl, nClass)
        if batch_u:
            semantic_scores_u = ret['semantic_scores_u']  # (Nu, nClass)
            embed_u = ret['embed_u']  # (Nu, C)

        # loss
        if not test:
            loss_inp = {}

            labels_l = batch_l['labels'].cuda()  # (Nl), long
            loss_inp['sup_loss'] = (semantic_scores_l, labels_l)
            if batch_u:
                idxos_1, idxos_2 = batch_u['idxos'][0].cuda(), batch_u['idxos'][1].cuda()   # (Nuo), long, Nuo = Nuo1 + ... + NuoB
                point_cnts_o = batch_u['point_cnts_o'].cuda()  # [Nuo1, ..., NuoB], int
                batch_offsets_u = batch_u['offsets'].cuda()  # (2B + 1), int
                loss_inp['unsup_loss'] = (embed_u, semantic_scores_u, idxos_1, idxos_2, point_cnts_o, batch_offsets_u)

            loss, loss_out, infos = loss_fn(loss_inp, step, model=model.module if hasattr(model, 'module') else model)

        # infos
        with torch.no_grad():
            preds = {}
            preds['semantic'] = semantic_scores_l

            if test:
                return preds

            p = semantic_scores_l.max(1)[1].cpu().numpy()
            gt = labels_l.cpu().numpy()
            i, u, target = utils.intersectionAndUnion(p, gt, cfg.classes, cfg.ignore_label)

            visual_dict = {}
            visual_dict['loss'] = loss
            for k, v in loss_out.items():
                visual_dict[k] = v[0]

            meter_dict = {}
            meter_dict['loss'] = (loss.item(), labels_l.shape[0])
            for k, v in loss_out.items():
                meter_dict[k] = (float(v[0]), v[1])

            meter_dict['intersection'] = (i, 1)
            meter_dict['union'] = (u, 1)
            meter_dict['target'] = (target, 1)

            return loss, preds, visual_dict, meter_dict

    def sup_loss_fn(scores, labels, step):
        if cfg.get('use_ce_thresh', False):
            def get_ce_thresh(cur_step):
                if cfg.get('use_descending_thresh', False):
                    start_thresh = 1.1
                    thresh = max(
                        cfg.ce_thresh,
                        start_thresh - (start_thresh - cfg.ce_thresh) * cur_step / cfg.ce_steps
                    )
                    return thresh
                elif cfg.get('use_ascending_thresh', False):
                    start_thresh = 1.0 / cfg.classes
                    thresh = min(
                        cfg.ce_thresh,
                        start_thresh + (cfg.ce_thresh - start_thresh) * cur_step / cfg.ce_steps
                    )
                else:
                    thresh = cfg.ce_thresh
                return thresh

            thresh = get_ce_thresh(step)
            temp_scores, _ = F.softmax(scores.detach(), dim=-1).max(1)
            mask = (temp_scores < thresh)
            if mask.sum() == 0:
                mask[torch.nonzero(labels >= 0)[0]] = True
            scores, labels = scores[mask], labels[mask]
        sup_loss = criterion(scores, labels)
        return sup_loss

    def unsup_loss_fn(embed_pos1, embed_pos2, embed_neg, pseudo_labels_pos1, pseudo_labels_neg, pconfs_pos2,
                      pos_idx1, pos_idx2, neg_idx):
        """loss on embed_pos1"""
        def run(embed, embed_neg, pseudo_labels, pseudo_labels_neg, pos_idx1, pos_idx2, neg_idx):
            neg = (embed @ embed_neg.T) / cfg.temp
            mask = torch.ones((embed.shape[0], embed_neg.shape[0]), dtype=torch.float32, device=embed.device)
            mask *= ((neg_idx.unsqueeze(0) != pos_idx1.unsqueeze(-1)).float() *
                     (neg_idx.unsqueeze(0) != pos_idx2.unsqueeze(-1)).float())
            pseudo_label_guidance = (pseudo_labels.unsqueeze(-1) != pseudo_labels_neg.unsqueeze(0)).float()
            mask *= pseudo_label_guidance
            neg = (torch.exp(neg) * mask).sum(-1)
            return neg

        pos = (embed_pos1 * embed_pos2.detach()).sum(-1, keepdim=True) / cfg.temp
        pos = torch.exp(pos).squeeze(-1)

        N = embed_neg.size(0)
        b = cfg.mem_batch_size
        neg = 0
        for i in range((N - 1) // b + 1):
            cur_embed_neg = embed_neg[i * b: (i + 1) * b]
            cur_pseudo_labels_neg = pseudo_labels_neg[i * b: (i + 1) * b]
            cur_neg_idx = neg_idx[i * b: (i + 1) * b]

            cur_neg = checkpoint.checkpoint(run, embed_pos1, cur_embed_neg, pseudo_labels_pos1, cur_pseudo_labels_neg,
                                            pos_idx2, pos_idx1, cur_neg_idx)
            neg += cur_neg

        eps = 1e-10
        unsup_loss = -torch.log(torch.clip(pos / torch.clip(pos + neg, eps), eps))

        confidence_guidance = (pconfs_pos2 >= cfg.conf_thresh).float()
        unsup_loss = (unsup_loss * confidence_guidance).sum() / torch.clip(confidence_guidance.sum(), eps)

        return unsup_loss

    def loss_fn(loss_inp, step, model=None):

        loss_out = {}
        infos = {}

        ## supervised loss
        semantic_scores_l, semantic_labels_l = loss_inp['sup_loss']

        sup_loss = sup_loss_fn(semantic_scores_l, semantic_labels_l, step)
        loss_out['sup_loss'] = (sup_loss, semantic_scores_l.shape[0])

        ## unsupervised loss
        if 'unsup_loss' in loss_inp:
            embed_u, semantic_scores_u, idxos1, idxos2, point_cnts_o, batch_offsets_u = loss_inp['unsup_loss']

            confs, pseudo_labels = F.softmax(semantic_scores_u, 1).max(1)    # (Nu), float/long, cuda

            # positive
            pos_sample_idx = get_pos_sample_idx(
                point_cnts_o, cfg.num_pos_sample, pseudo_labels[idxos1], num_classes=cfg.classes) # (bs * num_pos_sample), long, cuda

            embed_pos1, embed_pos2 = embed_u[idxos1][pos_sample_idx], embed_u[idxos2][pos_sample_idx]
            plabels_pos1, plabels_pos2 = pseudo_labels[idxos1][pos_sample_idx], pseudo_labels[idxos2][pos_sample_idx]
            confs_pos1, confs_pos2 = confs[idxos1][pos_sample_idx], confs[idxos2][pos_sample_idx]
            pos_idx1, pos_idx2 = idxos1[pos_sample_idx], idxos2[pos_sample_idx]

            # negtive
            embed_u1, embed_u2 = split_embed(embed_u, batch_offsets_u)  # (Nu1, C) / (Nu2, C), float, cuda
            plabels1, plabels2 = split_embed(pseudo_labels, batch_offsets_u)   # (Nu1) / (Nu2), long, cuda
            idx_u1, idx_u2 = split_embed(torch.arange(embed_u.shape[0], dtype=torch.long, device='cuda'), batch_offsets_u) # (Nu1) / (Nu2), long, cuda

            model.index_queue1[:] = -1
            model.index_queue2[:] = -1
            embed_neg1, plabels_neg1, neg_idx1 = get_neg_sample(embed_u1, idx_u1, plabels1,
                model.embed_queue1, model.index_queue1, model.queue_pointer1, model.valid_pointer1, cfg)
            embed_neg2, plabels_neg2, neg_idx2 = get_neg_sample(embed_u2, idx_u2, plabels2,
                model.embed_queue2, model.index_queue2, model.queue_pointer2, model.valid_pointer2, cfg)

            # unsup loss
            unsup_loss1 = unsup_loss_fn(embed_pos1, embed_pos2, embed_neg2, plabels_pos1, plabels_neg2, confs_pos2,
                                        pos_idx1, pos_idx2, neg_idx2)
            unsup_loss2 = unsup_loss_fn(embed_pos2, embed_pos1, embed_neg1, plabels_pos2, plabels_neg1, confs_pos1,
                                        pos_idx2, pos_idx1, neg_idx1)
            unsup_loss = unsup_loss1 + unsup_loss2

            loss_out['unsup_loss'] = (unsup_loss, embed_pos1.shape[0])

        ## total loss
        loss = cfg.loss_weight[0] * sup_loss
        if 'unsup_loss' in loss_out:
            loss += (cfg.loss_weight[1] * unsup_loss)

        return loss, loss_out, infos

    return model_fn
