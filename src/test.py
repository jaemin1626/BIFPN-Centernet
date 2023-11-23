from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os
import json
import cv2
import numpy as np
import time
from progress.bar import Bar
import torch
import torch.utils.data
# from external.nms import soft_nms
from lib.opts import opts
from lib.logger import Logger
from lib.utils.utils import AverageMeter
from lib.datasets.dataset_factory import dataset_factory
from lib.detectors.detector_factory import detector_factory
import sys
from lib.datasets.dataset_factory import get_dataset
class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

class PrefetchDataset(torch.utils.data.Dataset):
    def __init__(self, opt, dataset, pre_process_func):
        self.images = dataset.images
        self.load_image_func = dataset.coco.loadImgs
        self.img_dir = dataset.img_dir
        self.pre_process_func = pre_process_func
        self.opt = opt

    def __getitem__(self, index):
        img_id = self.images[index]
        img_info = self.load_image_func(ids=[img_id])[0]
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        image = cv2.imread(img_path)
        images, meta = {}, {}
        for scale in opt.test_scales:
            if opt.task == 'ddd':
                images[scale], meta[scale] = self.pre_process_func(
                    image, scale, img_info['calib'])
            else:
                images[scale], meta[scale] = self.pre_process_func(
                    image, scale)
        return img_id, {'images': images, 'image': image, 'meta': meta}

    def __len__(self):
        return len(self.images)


def prefetch_test(opt):
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str

    Dataset = dataset_factory[opt.dataset]
    opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
    print(opt)
    Logger(opt)
    Detector = detector_factory[opt.task]

    split = 'val' if not opt.trainval else 'test'
    dataset = Dataset(opt, split)
    detector = Detector(opt)

    data_loader = torch.utils.data.DataLoader(
        PrefetchDataset(opt, dataset, detector.pre_process),
        batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    results = {}
    num_iters = len(dataset)
    bar = Bar('{}'.format(opt.exp_id), max=num_iters)
    time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']
    avg_time_stats = {t: AverageMeter() for t in time_stats}
    for ind, (img_id, pre_processed_images) in enumerate(data_loader):
        ret = detector.run(pre_processed_images)
        results[img_id.numpy().astype(np.int32)[0]] = ret['results']
        Bar.suffix = '[{0}/{1}]|Tot: {total:} |ETA: {eta:} '.format(
            ind, num_iters, total=bar.elapsed_td, eta=bar.eta_td)
        for t in avg_time_stats:
            avg_time_stats[t].update(ret[t])
            Bar.suffix = Bar.suffix + '|{} {tm.val:.3f}s ({tm.avg:.3f}s) '.format(
                t, tm=avg_time_stats[t])
        bar.next()
    
    bar.finish()
    dataset.run_eval(results, opt.save_dir)
    # dataset.run_eval(results, opt.save_dir)


def test(opt):
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str

    Dataset = dataset_factory[opt.dataset]
    opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
    Logger(opt)
    Detector = detector_factory[opt.task]

    split = 'val' if not opt.trainval else 'test'
    dataset = Dataset(opt, split)
    detector = Detector(opt)

    results = {}
    num_iters = len(dataset)
    bar = Bar('{}'.format(opt.exp_id), max=num_iters)
    time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']
    avg_time_stats = {t: AverageMeter() for t in time_stats}
    for ind in range(num_iters):
        img_id = dataset.images[ind]
        img_info = dataset.coco.loadImgs(ids=[img_id])[0]
        img_path = os.path.join(dataset.img_dir, img_info['file_name'])

        if opt.task == 'ddd':
            ret = detector.run(img_path, img_info['calib'])
        else:
            ret = detector.run(img_path)

        results[img_id] = ret['results']

        Bar.suffix = '[{0}/{1}]|Tot: {total:} |ETA: {eta:} '.format(
            ind, num_iters, total=bar.elapsed_td, eta=bar.eta_td)
        for t in avg_time_stats:
            avg_time_stats[t].update(ret[t])
            Bar.suffix = Bar.suffix + \
                '|{} {:.3f} '.format(t, avg_time_stats[t].avg)
        bar.next()
    bar.finish()
    dataset.run_eval(results, opt.save_dir)

def train_inter_test_density(opt, val_loader, epoch):
    with HiddenPrints():
        os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str

        Dataset = dataset_factory[opt.dataset]
        opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
       
        Logger(opt)
        Detector = detector_factory[opt.task]

        split = 'val' if not opt.trainval else 'test'
        dataset = Dataset(opt, split)
        detector = Detector(opt)

        data_loader = torch.utils.data.DataLoader(
            PrefetchDataset(opt, dataset, detector.pre_process),
            batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
       
    results = {}
    num_iters = len(dataset)
    bar = Bar('{}'.format(opt.exp_id), max=num_iters)
    time_stats = []
    avg_time_stats = {t: AverageMeter() for t in time_stats}


    for ind, (img_id, pre_processed_images) in enumerate(data_loader):
        ret = detector.run(pre_processed_images)
        results[img_id.numpy().astype(np.int32)[0]] = ret
       
        Bar.suffix = '\033[32mval[{0}]: [{1}/{2}]|Tot: {total:} |ETA: {eta:}'.format(epoch,
            ind, num_iters, total=bar.elapsed_td, eta=bar.eta_td)
        for t in avg_time_stats:
            Bar.suffix = Bar.suffix + '|{} {tm.val:.3f}s '.format(t)
        bar.next()

    bar.finish()
    bar = Bar('{}'.format(opt.exp_id), max=num_iters)
    mae =[]
    for ind, batch in enumerate(val_loader):
        gt_count = batch['meta']['gt_det'].shape[1]
        res = results[ind+1]
        pr_count = 0

        for i in res['results'][1]:
            if i[4] >= 0.3:
                pr_count+=1

        # pr_count = torch.sum(res).item()
        mae_tmp = abs(gt_count - pr_count)
        mae.append(mae_tmp)

        middle = np.mean(mae)
        Bar.suffix = '\033[96mmae[{0}]: [{1}/{2}]|Tot: {total:} |ETA: {eta:} |MAE: {mae:.4f} '.format(epoch,
        ind, num_iters, total=bar.elapsed_td, eta=bar.eta_td, mae=middle)
        bar.next()


    mae = np.mean(mae)
    bar.finish()
    print('epoch ',epoch,' MAE : ',mae,'\033[37m')
   
    return mae

if __name__ == '__main__':
    opt = opts().parse()
    Dataset = get_dataset(opt.dataset, opt.task)
    
    val_loader = torch.utils.data.DataLoader(
        Dataset(opt, 'val'),
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    if opt.not_prefetch_test:
        test(opt)
    else:
        prefetch_test(opt)
    #train_inter_test_density(opt,val_loader=val_loader,epoch=100)