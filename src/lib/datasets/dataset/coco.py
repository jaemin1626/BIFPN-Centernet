from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pycocotools.coco as coco
from pycocotools.cocoeval import COCOeval
import numpy as np
import json
import os

import torch.utils.data as data

class COCO(data.Dataset):
    num_classes = 38
    default_resolution = [512, 512]
    mean = np.array([0.40789654, 0.44719302, 0.47026115],
                    dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.28863828, 0.27408164, 0.27809835],
                   dtype=np.float32).reshape(1, 1, 3)

    def __init__(self, opt, split):
        super(COCO, self).__init__()
        self.data_dir = os.path.join(opt.data_dir, 'coco')
        self.img_dir = os.path.join(self.data_dir, '{}2017'.format(split))
        if split == 'test':
            self.annot_path = os.path.join(
                self.data_dir, 'annotations',
                'image_info_test-dev2017.json').format(split)
        else:
            if opt.task == 'exdet':
                self.annot_path = os.path.join(
                    self.data_dir, 'annotations',
                    'instances_extreme_{}2017.json').format(split)
            else:
                self.annot_path = os.path.join(
                    self.data_dir, 'annotations',
                    'instances_{}2017.json').format(split)
        
        self.max_objs = 220
        # self.class_name = [
        #   '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
        #   'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
        #   'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
        #   'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
        #   'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
        #   'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
        #   'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass',
        #   'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
        #   'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
        #   'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
        #   'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
        #   'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
        #   'scissors', 'teddy bear', 'hair drier', 'toothbrush']
        #self.class_name = ['pest']
        
        self.class_name =  ['검거세미밤나방_유충','꽃노랑총재벌레_유충','담배가루이_유충','담배거세미나방_유충','담배나방_유충','도둑나방_유충','먹노린재_유충','묵화바둑명나방_유충','무잎벌_유충','배추좀나방_유충'\
                  ,'배추흰나비_유충','벼룩잎벌레_유충','복숭아혹진딧물_유충','비단노린재_유충','썩덩나무노린재_유충','열대거세미나방_유충','큰28점박이무당벌레_유충','톱다리개미허리노린재_유충','파밤나방_유충'\
                    '검거세미밤나방_성충','꽃노랑총재벌레_성충','담배가루이_성충','담배거세미나방_성충','담배나방_성충','도둑나방_성충','먹노린재_성충','묵화바둑명나방_성충','무잎벌_성충','배추좀나방_성충'\
                  ,'배추흰나비_성충','벼룩잎벌레_성충','복숭아혹진딧물_성충','비단노린재_성충','썩덩나무노린재_성충','열대거세미나방_성충','큰28점박이무당벌레_성충','톱다리개미허리노린재_성충','파밤나방_성충']
        
        self._valid_ids = [
          1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
          14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
          24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 
          34, 35, 36, 37, 38]

        self.cat_ids = {v: i for i, v in enumerate(self._valid_ids)}
        self.voc_color = [(v // 32 * 64 + 64, (v // 8) % 4 * 64, v % 8 * 32)
                          for v in range(1, self.num_classes + 1)]
        self._data_rng = np.random.RandomState(123)
        self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571],
                                 dtype=np.float32)
        self._eig_vec = np.array([
            [-0.58752847, -0.69563484, 0.41340352],
            [-0.5832747, 0.00994535, -0.81221408],
            [-0.56089297, 0.71832671, 0.41158938]
        ], dtype=np.float32)
        # self.mean = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 1, 3)
        # self.std = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 1, 3)

        self.split = split
        self.opt = opt

        print('==> initializing coco 2017 {} data.'.format(split))
        self.coco = coco.COCO(self.annot_path)
        self.images = self.coco.getImgIds()
        self.num_samples = len(self.images)

        print('Loaded {} {} samples'.format(split, self.num_samples))

    def _to_float(self, x):
        return float("{:.2f}".format(x))

    def convert_eval_format(self, all_bboxes):
        # import pdb; pdb.set_trace()
        detections = []
        for image_id in all_bboxes:
            for cls_ind in all_bboxes[image_id]:
                category_id = self._valid_ids[cls_ind - 1]
                for bbox in all_bboxes[image_id][cls_ind]:
                    bbox[2] -= bbox[0]
                    bbox[3] -= bbox[1]
                    score = bbox[4]
                    bbox_out = list(map(self._to_float, bbox[0:4]))

                    detection = {
                        "image_id": int(image_id),
                        "category_id": int(category_id),
                        "bbox": bbox_out,
                        "score": float("{:.2f}".format(score))
                    }
                    if len(bbox) > 5:
                        extreme_points = list(map(self._to_float, bbox[5:13]))
                        detection["extreme_points"] = extreme_points
                    detections.append(detection)
        return detections

    def __len__(self):
        return self.num_samples

    def save_results(self, results, save_dir):
        json.dump(self.convert_eval_format(results),
                  open('{}/results.json'.format(save_dir), 'w'))

    def run_eval(self, results, save_dir):
        # result_json = os.path.join(save_dir, "results.json")
        # detections  = self.convert_eval_format(results)
        # json.dump(detections, open(result_json, "w"))
        self.save_results(results, save_dir)
        coco_dets = self.coco.loadRes('{}/results.json'.format(save_dir))
        coco_eval = COCOeval(self.coco, coco_dets, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
