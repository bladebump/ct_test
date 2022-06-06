import argparse
import datetime
import numpy as np
import sys

import torch.dml
from torch.utils.data import DataLoader

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
        parser.add_argument('--num-workers', help='后台数据加载的核心数目', default=8, type=int)
        parser.add_argument('--batch-size', help='每轮训练的batch大小', default=32, type=int, )
        parser.add_argument('--epochs', help='训练的轮数', default=1, type=int, )
        parser.add_argument('--tb-prefix', default='luna',
                            help="Data prefix to use for Tensorboard run. Defaults to chapter.", )
        parser.add_argument('comment', help="Comment suffix for Tensorboard run.", nargs='?', default='dwlpt')

        self.cli_args = parser.parse_args(sys_argv)
        self.time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%S')

        self.use_dml = torch.dml.is_available()
        # self.use_dml = False
        self.device = torch.device('dml' if self.use_dml else 'cpu')

        self.model = self.init_module()
        self.optimizer = self.init_optimizer()

        self.total_training_samples_count = 0

    def init_module(self):
        model = LunaModule()
        if self.use_dml:
            log.info(f"using DML")
            model = model.to(self.device)
        return model

    def init_optimizer(self):
        # return SGD(self.model.parameters(), lr=0.001, momentum=0.99)
        return Adam(self.model.parameters())

    def init_train_dataloader(self):
        train_ds = LunaDatasets(val_stride=10, is_val_set_bool=False)
        batch_size = self.cli_args.batch_size
        if self.use_dml:
            batch_size *= 1

        train_dl = DataLoader(train_ds, batch_size=batch_size, num_workers=self.cli_args.num_workers,
                              pin_memory=self.use_dml)
        return train_dl

    def init_val_dataloader(self):
        val_ds = LunaDatasets(val_stride=10, is_val_set_bool=True)
        batch_size = self.cli_args.batch_size
        if self.use_dml:
            batch_size *= 1

        val_dl = DataLoader(val_ds, batch_size=batch_size, num_workers=self.cli_args.num_workers,
                            pin_memory=self.use_dml)
        return val_dl

    def main(self):
        log.info(f"开始 {type(self).__name__},{self.cli_args}")
        train_dl = self.init_train_dataloader()
        val_dl = self.init_val_dataloader()

        for epoch_ndx in range(1, self.cli_args.epochs + 1):
            trn_metrics_t = self.do_training(epoch_ndx, train_dl)
            self.log_metrics(epoch_ndx, '训练', trn_metrics_t)

            val_metrics_t = self.do_validation(epoch_ndx, val_dl)
            self.log_metrics(epoch_ndx, '验证', val_metrics_t)

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

        metrics_dict['correct/all'] = (pos_correct + neg_correct) / np.float(metrics_t.shape[1]) * 100
        metrics_dict['correct/neg'] = neg_correct / np.float(neg_count) * 100
        metrics_dict['correct/pos'] = pos_correct / np.float(pos_count) * 100

        log.info(
            "第{}轮 {:8} 损失是 {loss/all:.4f},正确率 {correct/all:-5.1f}%, ".format(epoch_ndx, mode_str, **metrics_dict)
        )

        log.info(
            "第{}轮 {:8} 损失是 {loss/all:.4f},正确率 {correct/all:-5.1f}%, ({neg_correct:} / {neg_count})".format(epoch_ndx,
                                                                                                           "阴性" + mode_str,
                                                                                                           neg_correct=neg_correct,
                                                                                                           neg_count=neg_count,
                                                                                                           **metrics_dict))

        log.info(
            "第{}轮 {:8} 损失是 {loss/all:.4f},正确率 {correct/all:-5.1f}%, ({pos_correct:} / {pos_count})".format(epoch_ndx,
                                                                                                           "阳性" + mode_str,
                                                                                                           pos_correct=pos_correct,
                                                                                                           pos_count=pos_count,
                                                                                                           **metrics_dict))


if __name__ == "__main__":
    LunaTrainingApp().main()
