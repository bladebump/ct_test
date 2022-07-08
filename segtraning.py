import argparse
import datetime
import hashlib
import logging
import os
import shutil
import sys

import numpy as np
import torch.cuda
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dstes import TrainingLuna2dSegmentationDataset, get_ct
from modle import UNetWrapper, SegmentationAugmentation
from util.util import enumerate_with_estimate

log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
# log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)

# METRICS_LABEL_NDX = 0
METRICS_LOSS_NDX = 1
# METRICS_FN_LOSS_NDX = 2
# METRICS_ALL_LOSS_NDX = 3

# METRICS_PTP_NDX = 4
# METRICS_PFN_NDX = 5
# METRICS_MFP_NDX = 6
METRICS_TP_NDX = 7
METRICS_FN_NDX = 8
METRICS_FP_NDX = 9

METRICS_SIZE = 10


class SegmentationTrainingApp:
    def __init__(self, sys_argv=None):
        if sys_argv is None:
            sys_argv = sys.argv[1:]

        parser = argparse.ArgumentParser()
        parser.add_argument('--batch-size',
                            help='Batch size to use for training',
                            default=16,
                            type=int,
                            )
        parser.add_argument('--num-workers',
                            help='Number of worker processes for background data loading',
                            default=8,
                            type=int,
                            )
        parser.add_argument('--epochs',
                            help='Number of epochs to train for',
                            default=1,
                            type=int,
                            )

        parser.add_argument('--augmented',
                            help="Augment the training data.",
                            action='store_true',
                            default=False,
                            )
        parser.add_argument('--augment-flip',
                            help="Augment the training data by randomly flipping the data left-right, up-down, and front-back.",
                            action='store_true',
                            default=False,
                            )
        parser.add_argument('--augment-offset',
                            help="Augment the training data by randomly offsetting the data slightly along the X and Y axes.",
                            action='store_true',
                            default=False,
                            )
        parser.add_argument('--augment-scale',
                            help="Augment the training data by randomly increasing or decreasing the size of the candidate.",
                            action='store_true',
                            default=False,
                            )
        parser.add_argument('--augment-rotate',
                            help="Augment the training data by randomly rotating the data around the head-foot axis.",
                            action='store_true',
                            default=False,
                            )
        parser.add_argument('--augment-noise',
                            help="Augment the training data by randomly adding noise to the data.",
                            action='store_true',
                            default=False,
                            )

        parser.add_argument('--tb-prefix',
                            default='p2ch13',
                            help="Data prefix to use for Tensorboard run. Defaults to chapter.",
                            )

        parser.add_argument('comment',
                            help="Comment suffix for Tensorboard run.",
                            nargs='?',
                            default='none',
                            )

        self.cli_args = parser.parse_args(sys_argv)
        self.time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%S')
        self.total_training_sample_count = 0
        self.trn_writer = None
        self.val_writer = None

        self.augmentation_dict = {}
        if self.cli_args.augmented or self.cli_args.augment_flip:
            self.augmentation_dict['flip'] = True
        if self.cli_args.augmented or self.cli_args.augment_offset:
            self.augmentation_dict['offset'] = 0.03
        if self.cli_args.augmented or self.cli_args.augment_scale:
            self.augmentation_dict['scale'] = 0.2
        if self.cli_args.augmented or self.cli_args.augment_rotate:
            self.augmentation_dict['rotate'] = True
        if self.cli_args.augmented or self.cli_args.augment_noise:
            self.augmentation_dict['noise'] = 25.0

        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda' if self.use_cuda else 'cpu')

        self.segmentation_model, self.augmentation_model = self.init_model()
        self.optimizer = self.init_optimizer()

    def init_model(self):
        segmentation_model = UNetWrapper(in_channels=7, n_classes=1, depth=3, wf=4, padding=True, batch_norm=True,
                                         up_mode='upconv')
        augmentation_model = SegmentationAugmentation(**self.augmentation_dict)
        if self.use_cuda:
            log.info("Using CUDA; {} devices.".format(torch.cuda.device_count()))
            if torch.cuda.device_count() > 1:
                segmentation_model = nn.DataParallel(segmentation_model)
                augmentation_model = nn.DataParallel(augmentation_model)
            segmentation_model = segmentation_model.to(self.device)
            augmentation_model = augmentation_model.to(self.device)
        return segmentation_model, augmentation_model

    def init_optimizer(self):
        return Adam(self.segmentation_model.parameters())

    def init_train_dl(self):
        train_ds = TrainingLuna2dSegmentationDataset(val_stride=10, is_val_set_bool=False, context_slices_count=3)

        batch_size = self.cli_args.batch_size
        if self.use_cuda:
            batch_size *= torch.cuda.device_count()

        train_dl = DataLoader(train_ds, batch_size=batch_size, num_workers=self.cli_args.num_workers,
                              pin_memory=self.use_cuda)

        return train_dl

    def init_val_dl(self):
        val_ds = TrainingLuna2dSegmentationDataset(val_stride=10, is_val_set_bool=True, context_slices_count=3)

        batch_size = self.cli_args.batch_size
        if self.use_cuda:
            batch_size *= torch.cuda.device_count()

        val_dl = DataLoader(val_ds, batch_size=batch_size, num_workers=self.cli_args.num_workers,
                            pin_memory=self.use_cuda)

        return val_dl

    def init_tensorboard_writers(self):
        if self.trn_writer is None:
            log_dir = os.path.join('runs', self.cli_args.tb_prefix, self.time_str)
            self.trn_writer = SummaryWriter(log_dir=log_dir + '-trn_cls-' + self.cli_args.comment)
            self.val_writer = SummaryWriter(log_dir=log_dir + '-val_cls-' + self.cli_args.comment)

    def main(self):
        log.info("Starting {}, {}".format(type(self).__name__, self.cli_args))
        train_dl = self.init_train_dl()
        val_dl = self.init_val_dl()

        best_score = 0.0
        self.validation_cadence = 5
        for epoch_ndx in range(1, self.cli_args.epochs + 1):
            log.info("Epoch {} of {}, {}/{} batches of size {}*{}".format(
                epoch_ndx,
                self.cli_args.epochs,
                len(train_dl),
                len(val_dl),
                self.cli_args.batch_size,
                (torch.cuda.device_count() if self.use_cuda else 1),
            ))

            trn_metrics_t = self.do_training(epoch_ndx, train_dl)
            self.log_metrics(epoch_ndx, 'trn', trn_metrics_t)

            if epoch_ndx == 1 or epoch_ndx % self.validation_cadence == 0:
                val_metrics_t = self.do_validation(epoch_ndx, val_dl)
                score = self.log_metrics(epoch_ndx, 'val', val_metrics_t)
                best_score = max(score, best_score)

                self.save_model('seg', epoch_ndx, score == best_score)

                self.log_images(epoch_ndx, 'trn', train_dl)
                self.log_images(epoch_ndx, 'val', val_dl)

        self.trn_writer.close()
        self.val_writer.close()

    def do_training(self, epoch_ndx, train_dl):
        trn_metrics_g = torch.zeros(METRICS_SIZE, len(train_dl.dataset), device=self.device)
        self.segmentation_model.train()
        train_dl.dataset.shuffle_sample()

        batch_iter = enumerate_with_estimate(train_dl, "E{} Training".format(epoch_ndx), start_ndx=train_dl.num_workers)
        for batch_ndx, batch_tup in batch_iter:
            self.optimizer.zero_grad()

            loss_val = self.compute_batch_loss(batch_ndx, batch_tup, train_dl.batch_size, trn_metrics_g)
            loss_val.backward()

            self.optimizer.step()
        self.total_training_sample_count += trn_metrics_g.size(1)

        return trn_metrics_g.to('cpu')

    def do_validation(self, epoch_ndx, val_dl):
        with torch.no_grad():
            val_metrics_g = torch.zeros(METRICS_SIZE, len(val_dl.dataset), device=self.device)
            self.augmentation_model.eval()
            batch_iter = enumerate_with_estimate(val_dl, "E{} Training".format(epoch_ndx),
                                                 start_ndx=val_dl.num_workers)
            for batch_ndx, batch_tup in batch_iter:
                self.compute_batch_loss(batch_ndx, batch_tup, val_dl.batch_size, val_metrics_g)
        return val_metrics_g.to('cpu')

    def compute_batch_loss(self, batch_ndx, batch_tup, batch_size, metrics_g, classification_threshold=0.5):
        input_t, label_t, series_list, _slice_ndx_list = batch_tup
        input_g, label_g = input_t.to(self.device, non_blocking=True), label_t.to(self.device, non_blocking=True)

        if self.segmentation_model.training and self.augmentation_dict:
            input_g, label_g = self.augmentation_model(input_g, label_g)

        prediction_g = self.segmentation_model(input_g)
        dice_loss_g = self.dice_loss(prediction_g, label_g)
        fn_loss_g = self.dice_loss(prediction_g * label_g, label_g)

        start_ndx = batch_ndx * batch_size
        end_ndx = start_ndx + input_t.size(0)

        with torch.no_grad():
            prediction_bool_g = (prediction_g[:, 0:1] > classification_threshold).to(torch.float32)
            tp = (prediction_bool_g * label_g).sum(dim=[1, 2, 3])
            fn = ((1 - prediction_bool_g) * label_g).sum(dim=[1, 2, 3])
            fp = (prediction_bool_g * ~label_g).sum(dim=[1, 2, 3])

            metrics_g[METRICS_LOSS_NDX, start_ndx:end_ndx] = dice_loss_g
            metrics_g[METRICS_TP_NDX, start_ndx:end_ndx] = tp
            metrics_g[METRICS_FN_NDX, start_ndx:end_ndx] = fn
            metrics_g[METRICS_FP_NDX, start_ndx:end_ndx] = fp

        return dice_loss_g.mean() + fn_loss_g.mean() * 8

    def dice_loss(self, prediction_g, label_g, epsilon=1):
        dice_label_g = label_g.sum(dim=[1, 2, 3])
        dice_prediction_g = prediction_g.sum(dim=[1, 2, 3])
        dice_correct_g = (prediction_g * label_g).sum(dim=[1, 2, 3])
        dice_ratio_g = (2 * dice_correct_g + epsilon) / (dice_prediction_g + dice_label_g + epsilon)
        return 1 - dice_ratio_g

    def log_metrics(self, epoch_ndx, mode_str, metrics_t):
        self.init_tensorboard_writers()
        log.info("E{} {}".format(epoch_ndx, type(self).__name__))
        metrics_a = metrics_t.detach().numpy()
        sum_a = metrics_a.sum(axis=1)
        assert np.isfinite(metrics_a).all()

        all_label_count = sum_a[METRICS_TP_NDX] + sum_a[METRICS_FN_NDX]
        metrics_dict = {}
        metrics_dict['loss/all'] = metrics_a[METRICS_LOSS_NDX].mean()
        metrics_dict['percent_all/tp'] = sum_a[METRICS_TP_NDX] / (all_label_count or 1) * 100
        metrics_dict['percent_all/fn'] = sum_a[METRICS_FN_NDX] / (all_label_count or 1) * 100
        metrics_dict['percent_all/fp'] = sum_a[METRICS_FP_NDX] / (all_label_count or 1) * 100

        precision = metrics_dict['pr/precision'] = sum_a[METRICS_TP_NDX] / (
                (sum_a[METRICS_TP_NDX] + sum_a[METRICS_FP_NDX]) or 1)
        recall = metrics_dict['pr/recall'] = sum_a[METRICS_TP_NDX] / (
                (sum_a[METRICS_TP_NDX] + sum_a[METRICS_FN_NDX]) or 1)
        metrics_dict['pr/f1_score'] = 2 * (precision * recall) / (precision + recall)
        log.info(("E{} {:8} "
                  + "{loss/all:.4f} loss, "
                  + "{pr/precision:.4f} precision, "
                  + "{pr/recall:.4f} recall, "
                  + "{pr/f1_score:.4f} f1 score"
                  ).format(
            epoch_ndx,
            mode_str,
            **metrics_dict,
        ))
        log.info(("E{} {:8} "
                  + "{loss/all:.4f} loss, "
                  + "{percent_all/tp:-5.1f}% tp, {percent_all/fn:-5.1f}% fn, {percent_all/fp:-9.1f}% fp"
                  ).format(
            epoch_ndx,
            mode_str + '_all',
            **metrics_dict,
        ))

        writer = getattr(self, mode_str + '_writer')
        prefix_str = 'seg_'
        for key, value in metrics_dict.items():
            writer.add_scalar(prefix_str + key, value, self.total_training_sample_count)

        writer.flush()

        score = metrics_dict['pr/recall']

        return score

    def log_images(self, epoch_ndx, mode_str, dl):
        self.segmentation_model.eval()

        images = sorted(dl.dataset.series_list)[:12]
        for series_ndx, series_uid in enumerate(images):
            ct = get_ct(series_uid)

            for slice_ndx in range(6):
                ct_ndx = slice_ndx * (ct.hu_a.shape[0] - 1) // 5
                sample_tup = dl.dataset.getitem_full_slice(series_uid, ct_ndx)

                ct_t, label_t, series_uid, ct_ndx = sample_tup

                input_g = ct_t.to(self.device).unsqueeze(0)
                label_g = label_t.to(self.device).unsqueeze(0)

                prediction_g = self.segmentation_model(input_g)[0]
                prediction_a = prediction_g.to('cpu').detach().numpy()[0] > 0.5
                label_a = label_g.cpu().numpy()[0][0] > 0.5

                ct_t[:-1, :, :] /= 2000
                ct_t[:-1, :, :] += 0.5

                ct_slice_a = ct_t[dl.dataset.context_slices_count].numpy()
                image_a = np.zeros((512, 512, 3), dtype=np.float)
                image_a[:, :, :] = ct_slice_a.reshape((512, 512, 1))
                image_a[:, :, 0] += prediction_a & (1 - label_a)
                image_a[:, :, 0] += (1 - prediction_a) & label_a
                image_a[:, :, 1] += ((1 - prediction_a) & label_a) * 0.5

                image_a[:, :, 1] += prediction_a & label_a
                image_a *= 0.5
                image_a.clip(0, 1, image_a)

                writer = getattr(self, mode_str + '_writer')
                writer.add_image(f'{mode_str}/{series_ndx}_prediction_{slice_ndx}', image_a,
                                 self.total_training_sample_count, dataformats='HWC', )

                if epoch_ndx == 1:
                    image_a = np.zeros((512, 512, 3), dtype=np.float32)
                    image_a[:, :, :] = ct_slice_a.reshape((512, 512, 1))
                    # image_a[:,:,0] += (1 - label_a) & lung_a # Red
                    image_a[:, :, 1] += label_a  # Green
                    # image_a[:,:,2] += neg_a  # Blue

                    image_a *= 0.5
                    image_a[image_a < 0] = 0
                    image_a[image_a > 1] = 1
                    writer.add_image(
                        '{}/{}_label_{}'.format(
                            mode_str,
                            series_ndx,
                            slice_ndx,
                        ),
                        image_a,
                        self.total_training_sample_count,
                        dataformats='HWC',
                    )
                writer.flush()

    def save_model(self, type_str, epoch_ndx, is_best=False):
        file_path = os.path.join(
            'data-unversioned',
            'part2',
            'models',
            self.cli_args.tb_prefix,
            '{}_{}_{}.{}.state'.format(
                type_str,
                self.time_str,
                self.cli_args.comment,
                self.total_training_sample_count,
            )
        )
        os.makedirs(os.path.dirname(file_path), mode=0o755, exist_ok=True)
        model = self.segmentation_model
        if isinstance(model, torch.nn.DataParallel):
            model = model.module

        state = {
            'sys_argv': sys.argv,
            'time': str(datetime.datetime.now()),
            'model_state': model.state_dict(),
            'model_name': type(model).__name__,
            'optimizer_state': self.optimizer.state_dict(),
            'optimizer_name': type(self.optimizer).__name__,
            'epoch': epoch_ndx,
            'totalTrainingSamples_count': self.total_training_sample_count,
        }
        torch.save(state, file_path)

        log.info("Saved model params to {}".format(file_path))

        if is_best:
            best_path = os.path.join(
                'data-unversioned', 'part2', 'models',
                self.cli_args.tb_prefix,
                f'{type_str}_{self.time_str}_{self.cli_args.comment}.best.state')
            shutil.copyfile(file_path, best_path)

            log.info("Saved model params to {}".format(best_path))

        with open(file_path, 'rb') as f:
            log.info("SHA1: " + hashlib.sha1(f.read()).hexdigest())
