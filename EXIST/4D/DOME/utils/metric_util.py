import numpy as np
from mmengine import MMLogger
import torch
import torch.distributed as dist
# logger = MMLogger.get_instance('genocc',distributed=dist.is_initialized())


class MeanIoU:

    def __init__(self,
                 class_indices,
                 ignore_label: int,
                 label_str,
                 name
                 # empty_class: int
        ):
        self.class_indices = class_indices
        self.num_classes = len(class_indices)
        self.ignore_label = ignore_label
        self.label_str = label_str
        self.name = name

    def reset(self) -> None:
        self.total_seen = torch.zeros(self.num_classes).cuda()
        self.total_correct = torch.zeros(self.num_classes).cuda()
        self.total_positive = torch.zeros(self.num_classes).cuda()

    def _after_step(self, outputs, targets, log_current=False):
        outputs = outputs[targets != self.ignore_label]
        targets = targets[targets != self.ignore_label]

        for i, c in enumerate(self.class_indices):
            self.total_seen[i] += torch.sum(targets == c).item()
            self.total_correct[i] += torch.sum((targets == c)
                                               & (outputs == c)).item()
            self.total_positive[i] += torch.sum(outputs == c).item()

    def _after_epoch(self):
        if dist.is_initialized():
            dist.all_reduce(self.total_seen)
            dist.all_reduce(self.total_correct)
            dist.all_reduce(self.total_positive)

        ious = []

        for i in range(self.num_classes):
            if self.total_seen[i] == 0:
                ious.append(1)
            else:
                cur_iou = self.total_correct[i] / (self.total_seen[i]
                                                   + self.total_positive[i]
                                                   - self.total_correct[i])
                ious.append(cur_iou.item())

        miou = np.mean(ious)
        logger = MMLogger.get_current_instance()
        logger.info(f'Validation per class iou {self.name}:')
        for iou, label_str in zip(ious, self.label_str):
            logger.info('%s : %.2f%%' % (label_str, iou * 100))
        
        return miou * 100


class multi_step_MeanIou:
    def __init__(self,
                 class_indices,
                 ignore_label: int,
                 label_str,
                 name,
                 times=1):
        self.class_indices = class_indices
        self.num_classes = len(class_indices)
        self.ignore_label = ignore_label
        self.label_str = label_str
        self.name = name
        self.times = times
        
    def reset(self) -> None:
        self.total_seen = torch.zeros(self.times, self.num_classes).cuda()
        self.total_correct = torch.zeros(self.times, self.num_classes).cuda()
        self.total_positive = torch.zeros(self.times, self.num_classes).cuda()
        self.current_seen = torch.zeros(self.times, self.num_classes).cuda()
        self.current_correct = torch.zeros(self.times, self.num_classes).cuda()
        self.current_positive = torch.zeros(self.times, self.num_classes).cuda()
    
    def _after_step(self, outputses, targetses,log_current=False):
        
        assert outputses.shape[1] == self.times, f'{outputses.shape[1]} != {self.times}'
        assert targetses.shape[1] == self.times, f'{targetses.shape[1]} != {self.times}'
        mious = []
        for t in range(self.times):
            ious = []
            outputs = outputses[:,t, ...][targetses[:,t, ...] != self.ignore_label].cuda()
            targets = targetses[:,t, ...][targetses[:,t, ...] != self.ignore_label].cuda()
            for j, c in enumerate(self.class_indices):
                self.total_seen[t, j] += torch.sum(targets == c).item()
                self.total_correct[t, j] += torch.sum((targets == c)
                                                      & (outputs == c)).item()
                self.total_positive[t, j] += torch.sum(outputs == c).item()
                if log_current:
                    current_seen = torch.sum(targets == c).item()
                    current_correct = torch.sum((targets == c)& (outputs == c)).item()
                    current_positive = torch.sum(outputs == c).item()
                    if current_seen == 0:
                        ious.append(1)
                    else:
                        cur_iou = current_correct / (current_seen+current_positive-current_correct)
                        ious.append(cur_iou)
            if log_current:
                miou = np.mean(ious)
                logger = MMLogger.get_current_instance()#distributed=dist.is_initialized())
                logger.info(f'current:: per class iou {self.name} at time {t}:')
                for iou, label_str in zip(ious, self.label_str):
                    logger.info('%s : %.2f%%' % (label_str, iou * 100))
                logger.info(f'mIoU {self.name} at time {t}: %.2f%%' % (miou * 100))
                mious.append(miou * 100)
        m_miou=np.mean(mious)
        # mious=torch.tensor(mious).cuda()
        return mious, m_miou
    
    def _after_epoch(self):
        logger = MMLogger.get_current_instance()#distributed=dist.is_initialized())
        if dist.is_initialized():
            dist.all_reduce(self.total_seen)
            dist.all_reduce(self.total_correct)
            dist.all_reduce(self.total_positive)
            logger.info(f'_after_epoch::total_seen: {self.total_seen.sum()}')
            logger.info(f'_after_epoch::total_correct: {self.total_correct.sum()}')
            logger.info(f'_after_epoch::total_positive: {self.total_positive.sum()}')
        mious = []
        for t in range(self.times):
            ious = []
            for i in range(self.num_classes):
                if self.total_seen[t, i] == 0:
                    ious.append(1)
                else:
                    cur_iou = self.total_correct[t, i] / (self.total_seen[t, i]
                                                          + self.total_positive[t, i]
                                                          - self.total_correct[t, i])
                    ious.append(cur_iou.item())
            miou = np.mean(ious)
            logger.info(f'per class iou {self.name} at time {t}:')
            for iou, label_str in zip(ious, self.label_str):
                logger.info('%s : %.2f%%' % (label_str, iou * 100))
            logger.info(f'mIoU {self.name} at time {t}: %.2f%%' % (miou * 100))
            mious.append(miou * 100)
        return mious, np.mean(mious)