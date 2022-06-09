import argparse
import datetime
import os.path

import numpy as np
import sys

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dstes import LunaDatasets
from modle import LunaModule
from torch import nn
from torch.optim import SGD, Adam

from util.util import enumerate_with_estimate
from applog import logging

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

METRICS_LABEL_NDX = 0
METRICS_PRED_NDX = 1
METRICS_LOSS_NDX = 2
METRICS_SIZE = 3


class LunaTrainingApp:
    def __init__(self, sys_argv=None):
        if sys_argv is None:
            sys_argv = sys.argv[1:]
        parser = argparse.ArgumentParser()
        parser.add_argument('--num-workers', help='后台数据加载的核心数目', default=2, type=int)
        parser.add_argument('--batch-size', help='每轮训练的batch大小', default=64, type=int, )
        parser.add_argument('--epochs', help='训练的轮数', default=1, type=int, )
        parser.add_argument('--tb-prefix', default='luna',
                            help="Data prefix to use for Tensorboard run. Defaults to chapter.", )
        parser.add_argument('comment', help="Comment suffix for Tensorboard run.", nargs='?', default='dwlpt')
        parser.add_argument('--conti', help="是否通过保存的节点继续", default=False, type=bool)

        self.cli_args = parser.parse_args(sys_argv)
        self.time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%S')

        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda' if self.use_cuda else 'cpu')

        self.trn_writer = None
        self.val_writer = None
        self.total_training_samples_count = 0

        self.model = self.init_module()
        self.optimizer = self.init_optimizer()

        self.save_path = os.path.join("checkpt", "luna.pth")

        if self.cli_args.conti:
            self.load()

    def init_module(self):
        model = LunaModule()
        if self.use_cuda:
            log.info(f"using cuda")
            model = model.to(self.device)
        return model

    def init_optimizer(self):
        # return SGD(self.model.parameters(), lr=0.001, momentum=0.99)
        return Adam(self.model.parameters())

    def init_train_dataloader(self):
        train_ds = LunaDatasets(val_stride=10, is_val_set_bool=False)
        batch_size = self.cli_args.batch_size
        if self.use_cuda:
            batch_size *= torch.cuda.device_count()

        train_dl = DataLoader(train_ds, batch_size=batch_size, num_workers=self.cli_args.num_workers,
                              pin_memory=self.use_cuda)
        return train_dl

    def init_val_dataloader(self):
        val_ds = LunaDatasets(val_stride=10, is_val_set_bool=True)
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
        log.info(f"开始 {type(self).__name__},{self.cli_args}")
        train_dl = self.init_train_dataloader()
        val_dl = self.init_val_dataloader()

        for epoch_ndx in range(1, self.cli_args.epochs + 1):
            trn_metrics_t = self.do_training(epoch_ndx, train_dl)
            self.log_metrics(epoch_ndx, 'trn', trn_metrics_t)

            val_metrics_t = self.do_validation(epoch_ndx, val_dl)
            self.log_metrics(epoch_ndx, 'val', val_metrics_t)

        if hasattr(self, 'trn_writer'):
            self.trn_writer.close()
            self.val_writer.close()

    def do_training(self, epoch_ndx, train_dl):
        self.model.train()
        trn_metrics_g = torch.zeros(METRICS_SIZE, len(train_dl.dataset), device=self.device)

        batch_iter = enumerate_with_estimate(train_dl, "第{}轮训练".format(epoch_ndx), start_ndx=train_dl.num_workers)

        for batch_ndx, batch_tup in batch_iter:
            self.optimizer.zero_grad()
            loss_var = self.compute_batch_loss(batch_ndx, batch_tup, train_dl.batch_size, trn_metrics_g)
            loss_var.backward()
            self.optimizer.step()

        self.total_training_samples_count += len(train_dl.dataset)

        return trn_metrics_g.to('cpu')

    def do_validation(self, epoch_ndx, val_dl):
        with torch.no_grad():
            self.model.eval()
            val_metrics_g = torch.zeros(METRICS_SIZE, len(val_dl.dataset), device=self.device)

            batch_iter = enumerate_with_estimate(val_dl, f"第{epoch_ndx}轮测试", start_ndx=val_dl.num_workers)

            for batch_ndx, batch_tup in batch_iter:
                self.compute_batch_loss(batch_ndx, batch_tup, val_dl.batch_size, val_metrics_g)

        return val_metrics_g.to('cpu')

    def compute_batch_loss(self, batch_ndx, batch_tup, batch_size, metrics_g):
        input_t, label_t, _series_list, _center_list = batch_tup
        input_g = input_t.to(self.device, non_blocking=True)
        label_g = label_t.to(self.device, non_blocking=True)

        logits_g, probability_g = self.model(input_g)

        loss_func = nn.CrossEntropyLoss(reduction='none')
        loss_g = loss_func(logits_g, label_g[:, 1])

        start_ndx = batch_ndx * batch_size
        end_ndx = start_ndx + label_t.size(0)

        metrics_g[METRICS_LABEL_NDX, start_ndx:end_ndx] = label_g[:, 1].detach()
        metrics_g[METRICS_PRED_NDX, start_ndx:end_ndx] = probability_g[:, 1].detach()
        metrics_g[METRICS_LOSS_NDX, start_ndx:end_ndx] = loss_g.detach()
        return loss_g.mean()

    def log_metrics(self, epoch_ndx, mode_str, metrics_t, classification_threshold=0.5):
        self.init_tensorboard_writers()
        log.info(f"第{epoch_ndx}轮 {type(self).__name__}")
        neg_label_mask = metrics_t[METRICS_LABEL_NDX] <= classification_threshold
        neg_pred_mask = metrics_t[METRICS_PRED_NDX] <= classification_threshold

        pos_label_mask = ~neg_label_mask
        pos_pred_mask = ~neg_pred_mask

        neg_count = int(neg_label_mask.sum())
        pos_count = int(pos_label_mask.sum())

        neg_correct = int((neg_label_mask & neg_pred_mask).sum())
        pos_correct = int((pos_label_mask & pos_pred_mask).sum())

        metrics_dict = {}
        metrics_dict['loss/all'] = metrics_t[METRICS_LOSS_NDX].mean()
        metrics_dict['loss/neg'] = metrics_t[METRICS_LOSS_NDX, neg_label_mask].mean()
        metrics_dict['loss/pos'] = metrics_t[METRICS_LOSS_NDX, pos_label_mask].mean()

        metrics_dict['correct/all'] = (pos_correct + neg_correct) / float(metrics_t.shape[1]) * 100
        metrics_dict['correct/neg'] = neg_correct / float(neg_count) * 100
        metrics_dict['correct/pos'] = pos_correct / float(pos_count) * 100

        log.info(
            "第{}轮 {:8} 损失是 {loss/all:.4f},正确率 {correct/all:-5.1f}%, ".format(epoch_ndx, mode_str, **metrics_dict)
        )

        log.info(
            "第{}轮 {:8} 损失是 {loss/neg:.4f},正确率 {correct/neg:-5.1f}%, ({neg_correct:} / {neg_count})".format(epoch_ndx,
                                                                                                           "阴性" + mode_str,
                                                                                                           neg_correct=neg_correct,
                                                                                                           neg_count=neg_count,
                                                                                                           **metrics_dict))

        log.info(
            "第{}轮 {:8} 损失是 {loss/pos:.4f},正确率 {correct/pos:-5.1f}%, ({pos_correct:} / {pos_count})".format(epoch_ndx,
                                                                                                           "阳性" + mode_str,
                                                                                                           pos_correct=pos_correct,
                                                                                                           pos_count=pos_count,
                                                                                                           **metrics_dict))

        writer = getattr(self, mode_str + '_writer')
        for key, value in metrics_dict.items():
            writer.add_scalar(key, value, self.total_training_samples_count)

        writer.add_pr_curve('pr', metrics_t[METRICS_LABEL_NDX], metrics_t[METRICS_PRED_NDX],
                            self.total_training_samples_count)
        bins = [x / 50.0 for x in range(51)]
        neg_hist_mask = neg_label_mask & (metrics_t[METRICS_PRED_NDX] > 0.01)
        pos_hist_mask = pos_label_mask & (metrics_t[METRICS_PRED_NDX] > 0.99)

        if neg_hist_mask.any():
            writer.add_histogram('is_neg', metrics_t[METRICS_PRED_NDX, neg_hist_mask],
                                 self.total_training_samples_count, bins=bins)
        if pos_hist_mask.any():
            writer.add_histogram('is_pos', metrics_t[METRICS_PRED_NDX, pos_hist_mask],
                                 self.total_training_samples_count, bins=bins)

    def save(self):
        torch.save(dict(model=self.model.state_dict(), optimizer=self.optimizer.state_dict()), self.save_path)

    def load(self):
        if os.path.exists(self.save_path):
            save_dict = torch.load(self.save_path, map_location=self.device)
            self.model.load_state_dict(save_dict['model'])
            self.optimizer.load_state_dict(save_dict['optimizer'])


if __name__ == "__main__":
    m = LunaTrainingApp()
    try:
        m.main()
    except KeyboardInterrupt:
        m.save()
