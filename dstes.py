import copy
import csv
import functools
import logging
import math
import pathlib
import datetime
import random

import SimpleITK as sitk
from collections import namedtuple

import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

from util.util import XyzTuple, xyz2irc, enumerate_with_estimate
from util.disk import get_cache

log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
# log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)

CandidateInfoTuple = namedtuple(
    'CandidateInfoTuple',
    'isNodule_bool,diameter_mm,series_uid,center_xyz',
)

root_dir_path = pathlib.Path('data')
raw_cache = get_cache('raw')


@functools.lru_cache(1)
def get_candidate_info_list(require_on_disk_bool=True):
    mhd_list = root_dir_path.glob('subset*/*.mhd')
    present_on_dist = {p.parts[-1][:-4] for p in mhd_list}

    diameter_dict = {}
    with open(root_dir_path.joinpath('annotations.csv'), 'r') as f:
        for row in list(csv.reader(f))[1:]:
            series_uid = row[0]
            annotation_center_xyz = tuple([float(x) for x in row[1:4]])
            annotation_diameter_mm = float(row[4])
            diameter_dict.setdefault(series_uid, []).append((annotation_center_xyz, annotation_diameter_mm))

    candidate_info_list = []
    with open(root_dir_path.joinpath('candidates.csv'), 'r') as f:
        for row in list(csv.reader(f))[1:]:
            series_uid = row[0]

            if series_uid not in present_on_dist and require_on_disk_bool:
                continue

            is_nodule_bool = bool(int(row[4]))
            candidate_center_xyz = tuple([float(x) for x in row[1:4]])

            candidate_diameter_mm = 0.0
            for annotation_tup in diameter_dict.get(series_uid, []):
                annotation_center_xyz, annotation_diameter_mm = annotation_tup
                for i in range(3):
                    delta_mm = abs(candidate_center_xyz[i] - annotation_center_xyz[i])
                    if delta_mm > annotation_diameter_mm / 4:
                        break
                else:
                    candidate_diameter_mm = annotation_diameter_mm
                    break
            candidate_info_list.append(
                CandidateInfoTuple(
                    is_nodule_bool, candidate_diameter_mm, series_uid, candidate_center_xyz
                )
            )

    candidate_info_list.sort(reverse=True)
    return candidate_info_list


class Ct:
    def __init__(self, series_uid):
        mhd_path = next(root_dir_path.glob('subset*/{}.mhd'.format(series_uid)))
        ct_mhd = sitk.ReadImage(str(mhd_path))
        ct_a = np.array(sitk.GetArrayFromImage(ct_mhd), dtype=np.float32)
        ct_a.clip(-1000, 1000, ct_a)

        self.series_uid = series_uid
        self.hu_a = ct_a

        self.origin_xyz = XyzTuple(*ct_mhd.GetOrigin())
        self.vx_size_xyz = XyzTuple(*ct_mhd.GetSpacing())
        self.direction_a = np.array(ct_mhd.GetDirection()).reshape(3, 3)

    def get_raw_candidate(self, center_xyz, width_irc):
        center_irc = xyz2irc(center_xyz, self.origin_xyz, self.vx_size_xyz, self.direction_a)
        slice_list = []
        for axis, center_val in enumerate(center_irc):
            start_ndx = int(round(center_val - width_irc[axis] / 2))
            end_ndx = int(start_ndx + width_irc[axis])

            assert 0 <= center_val < self.hu_a.shape[axis], repr(
                [self.series_uid, center_xyz, self.origin_xyz, self.vx_size_xyz, center_irc])
            if start_ndx < 0:
                start_ndx = 0
                end_ndx = int(width_irc[axis])
            if end_ndx > self.hu_a.shape[axis]:
                end_ndx = self.hu_a.shape[axis]
                start_ndx = int(self.hu_a.shape[axis] - width_irc[axis])
            slice_list.append(slice(start_ndx, end_ndx))
        ct_chunk = self.hu_a[tuple(slice_list)]
        return ct_chunk, center_irc


@functools.lru_cache(1, typed=True)
def get_ct(series_uid):
    return Ct(series_uid)


@raw_cache.memoize(typed=True)
def get_ct_raw_candidate(series_uid, center_xyz, width_irc):
    ct = get_ct(series_uid)
    ct_chunk, center_irc = ct.get_raw_candidate(center_xyz, width_irc)
    return ct_chunk, center_irc


def get_ct_augmented_candidate(augmentation_dict, series_uid, center_xyz, width_irc, use_cache=True):
    if use_cache:
        ct_chunk, center_irc = get_ct_raw_candidate(series_uid, center_xyz, width_irc)
    else:
        ct = get_ct(series_uid)
        ct_chunk, center_irc = ct.get_raw_candidate(center_xyz, width_irc)
    ct_t = torch.tensor(ct_chunk).unsqueeze(0).unsqueeze(0).to(torch.float32)

    transform_t = torch.eye(4)
    for i in range(3):
        if 'flip' in augmentation_dict:
            if random.random() > 0.5:
                transform_t[i, i] *= -1

        if 'offset' in augmentation_dict:
            offset_float = augmentation_dict['offset']
            random_float = (random.random() * 2 - 1)
            transform_t[i, 3] = offset_float * random_float

        if 'scale' in augmentation_dict:
            scale_float = augmentation_dict['scale']
            random_float = (random.random() * 2 - 1)
            transform_t[i, i] *= 1.0 + scale_float * random_float

    if 'rotate' in augmentation_dict:
        angle_rad = random.random() * math.pi * 2
        s = math.sin(angle_rad)
        c = math.cos(angle_rad)

        rotation_t = torch.tensor([
            [c, -s, 0, 0],
            [s, c, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        transform_t @= rotation_t

    affine_t = F.affine_grid(transform_t[:3].unsqueeze(0).to(torch.float32), ct_t.size(), align_corners=False)
    augmented_chunk = F.grid_sample(ct_t, affine_t, padding_mode='border', align_corners=False).to('cpu')
    if 'noise' in augmentation_dict:
        noise_t = torch.randn_like(augmented_chunk)
        noise_t *= augmented_chunk['noise']
        augmented_chunk += noise_t

    return augmented_chunk[0], center_irc


class LunaDatasets(Dataset):
    def __init__(self, val_stride=0, is_val_set_bool=None, series_uid=None, sortby_str='random', ratio_int=0,
                 augmentation_dict=None, candidate_info_list=None):
        self.ratio_int = ratio_int
        self.augmentation_dict = augmentation_dict

        if candidate_info_list:
            self.candidate_info_list = copy.copy(candidate_info_list)
            self.use_cache = False
        else:
            self.candidate_info_list = copy.copy(get_candidate_info_list())
            self.use_cache = True

        if series_uid:
            self.candidate_info_list = [x for x in self.candidate_info_list if x.series_uid == series_uid]

        if sortby_str == 'random':
            random.shuffle(self.candidate_info_list)
        elif sortby_str == 'series_uid':
            self.candidate_info_list.sort(key=lambda x: (x.series_uid, x.center_xyz))
        elif sortby_str == 'label_and_size':
            pass
        else:
            raise Exception("未知的数据集排序方式:" + repr(sortby_str))


        if is_val_set_bool:
            assert val_stride > 0, val_stride
            self.candidate_info_list = self.candidate_info_list[::val_stride]
            assert self.candidate_info_list
        elif val_stride > 0:
            del self.candidate_info_list[::val_stride]
            assert self.candidate_info_list

        self.negative_list = [nt for nt in self.candidate_info_list if not nt.isNodule_bool]
        self.pos_list = [nt for nt in self.candidate_info_list if nt.isNodule_bool]

        log.info("{!r}: {} {} 样本, {} neg, {} pos, {} ratio".format(
            self,
            len(self.candidate_info_list),
            "验证" if is_val_set_bool else "训练",
            len(self.negative_list),
            len(self.pos_list),
            '{}:1'.format(self.ratio_int) if self.ratio_int else 'unbalanced'
        ))

    def shuffle_samples(self):
        if self.ratio_int:
            random.shuffle(self.negative_list)
            random.shuffle(self.pos_list)

    def __len__(self):
        if self.ratio_int:
            return 20000
        else:
            return len(self.candidate_info_list)

    def __getitem__(self, ndx):
        if self.ratio_int:
            pos_ndx = ndx

            if ndx % (self.ratio_int + 1):
                neg_ndx = ndx - 1 - pos_ndx
                neg_ndx %= len(self.negative_list)
                candidate_info_tup = self.negative_list[neg_ndx]
            else:
                pos_ndx %= len(self.pos_list)
                candidate_info_tup = self.pos_list[pos_ndx]
        else:
            candidate_info_tup = self.candidate_info_list[ndx]
        width_irc = (32, 48, 48)

        if self.augmentation_dict:
            candidate_t, center_irc = get_ct_augmented_candidate(self.augmentation_dict, candidate_info_tup.series_uid,
                                                                 candidate_info_tup.center_xyz, width_irc,
                                                                 self.use_cache)
        elif self.use_cache:
            candidate_a, center_irc = get_ct_raw_candidate(
                candidate_info_tup.series_uid,
                candidate_info_tup.center_xyz,
                width_irc
            )
            candidate_t = torch.from_numpy(candidate_a)
            candidate_t = candidate_t.to(torch.float32)
            candidate_t = candidate_t.unsqueeze(0)
        else:
            ct = get_ct(candidate_info_tup.series_uid)
            candidate_a, center_irc = ct.get_raw_candidate(
                candidate_info_tup.center_xyz,
                width_irc,
            )
            candidate_t = torch.from_numpy(candidate_a).to(torch.float32)
            candidate_t = candidate_t.unsqueeze(0)

        pos_t = torch.tensor([not candidate_info_tup.isNodule_bool, candidate_info_tup.isNodule_bool], dtype=torch.long)
        return candidate_t, pos_t, candidate_info_tup.series_uid, torch.tensor(center_irc)
