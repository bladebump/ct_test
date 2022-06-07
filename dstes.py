import copy
import csv
import functools
import pathlib
import datetime
import random

import SimpleITK as sitk
from collections import namedtuple

import numpy as np
import torch
from torch.utils.data import Dataset

from util.util import XyzTuple, xyz2irc, enumerate_with_estimate
from util.disk import get_cache

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


class LunaDatasets(Dataset):
    def __init__(self, val_stride=0, is_val_set_bool=None, series_uid=None, shuffle=False):
        self.candidate_info_list = copy.copy(get_candidate_info_list())

        if shuffle:
            random.shuffle(self.candidate_info_list)

        if series_uid:
            self.candidate_info_list = [x for x in self.candidate_info_list if x.series_uid == series_uid]

        if is_val_set_bool:
            assert val_stride > 0, val_stride
            self.candidate_info_list = self.candidate_info_list[::val_stride]
            assert self.candidate_info_list
        elif val_stride > 0:
            del self.candidate_info_list[::val_stride]
            assert self.candidate_info_list

    def __len__(self):
        return len(self.candidate_info_list)

    def __getitem__(self, ndx):
        candidate_info_tup = self.candidate_info_list[ndx]
        width_irc = (32, 48, 48)
        candidate_a, center_irc = get_ct_raw_candidate(
            candidate_info_tup.series_uid,
            candidate_info_tup.center_xyz,
            width_irc
        )
        candidate_t = torch.from_numpy(candidate_a)
        candidate_t = candidate_t.to(torch.float32)
        candidate_t = candidate_t.unsqueeze(0)

        pos_t = torch.tensor([not candidate_info_tup.isNodule_bool, candidate_info_tup.isNodule_bool], dtype=torch.long)
        return candidate_t, pos_t, candidate_info_tup.series_uid, torch.tensor(center_irc)


def test1():
    start = datetime.datetime.now()
    ds = LunaDatasets()
    iter = enumerate_with_estimate(ds, "test")
    for i, data in iter:
        if data[0].shape[1] != 32:
            print(data[0].shape, data[1], data[2], data[3])
    end = datetime.datetime.now()
    print(end - start)


def test2():
    ds = LunaDatasets()
    i, data = next(enumerate(ds))
    input_t, label_t, _series_list, _center_list = batch_tup = data
    print(label_t)


if __name__ == "__main__":
    test1()
