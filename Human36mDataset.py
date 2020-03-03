import os
import json
import glob
from scipy.io import loadmat
import cv2
import numpy as np
import random

import torch
import torch.nn as nn

file_path = os.path.dirname(os.path.realpath(__file__))
cwd = os.getcwd()

if file_path != cwd:
    from .utils.tools import pose_world_to_cam, pose_cam_to_pixel, draw_skeleton, normalize_3d_pose
    from .config import H36mDatasetCfg
    from .utils.augment import augment
else:
    from utils.tools import pose_world_to_cam, pose_cam_to_pixel, draw_skeleton, normalize_3d_pose
    from config import H36mDatasetCfg
    from utils.augment import augment


class GenerateHeatmap:
    def __init__(self, output_res, num_parts):
        self.output_res = output_res
        self.num_parts = num_parts
        sigma = self.output_res[0] / 64
        self.sigma = sigma
        size = 6 * sigma + 3
        x = np.arange(0, size, 1, float)
        y = x[:, np.newaxis]
        x0, y0 = 3 * sigma + 1, 3 * sigma + 1
        self.g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

    def __call__(self, keypoints):
        hms = np.zeros(shape=(self.num_parts, self.output_res[1], self.output_res[0]), dtype=np.float32)
        sigma = self.sigma
        for idx, pt in enumerate(keypoints):
            if pt[0] > 0:
                x, y = int(pt[0]), int(pt[1])
                if x < 0 or y < 0 or x >= self.output_res[0] or y >= self.output_res[1]:
                    continue
                ul = int(x - 3 * sigma - 1), int(y - 3 * sigma - 1)
                br = int(x + 3 * sigma + 2), int(y + 3 * sigma + 2)

                c, d = max(0, -ul[0]), min(br[0], self.output_res[0]) - ul[0]
                a, b = max(0, -ul[1]), min(br[1], self.output_res[1]) - ul[1]

                cc, dd = max(0, ul[0]), min(br[0], self.output_res[0])
                aa, bb = max(0, ul[1]), min(br[1], self.output_res[1])
                hms[idx, aa:bb, cc:dd] = np.maximum(hms[idx, aa:bb, cc:dd], self.g[a:b, c:d])
        return hms


class Human36mDataset(torch.utils.data.Dataset):
    def __init__(self, data_root, protocols=[1, 5, 6, 7, 8], is_train=True, heatmap=False, config=None):
        super(Human36mDataset, self).__init__()
        self.protocols = protocols
        self.is_train = is_train
        self.data_root = data_root
        self.heatmap = heatmap
        if config is not None:
            self.config = config
        else:
            self.config = H36mDatasetCfg
        if self.heatmap:
            self.generate_heatmap = GenerateHeatmap(self.config['out_res'], 17)
        self.is_aug = self.config['is_aug']
        self.with_bbox = self.config['with_bbox']
        self.images_root = os.path.join(self.data_root, 'images')
        self.anno_root = os.path.join(self.data_root, 'annotations')

        self.skeleton = ((0, 7), (7, 8), (8, 9), (9, 10), (8, 11), (11, 12), (12, 13), (8, 14),
                         (14, 15), (15, 16), (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6))
        self.flip_pair = ((1, 4), (2, 5), (3, 6), (14, 11), (15, 12), (16, 13))
        self.joint_name = ('Pelvis', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Torso', 'Neck',
                           'Nose', 'Head', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Shoulder', 'R_Elbow', 'R_Wrist',
                           'Thorax')

        # self.camera_params = self.load_cam_params(self.anno_root, self.protocols)
        self.seqs_folder = []

        for subject in protocols:
            self.seqs_folder.extend([os.path.basename(seq_path)
                                     for seq_path in
                                     sorted(glob.glob(os.path.join(self.images_root, 's_%02d_*' % subject)))])

        self.meta_data = []
        self.index_mapping = {}
        index = 0
        for idx, seq_folder in enumerate(self.seqs_folder):
            if self.with_bbox:
                self.meta_data.append(loadmat(os.path.join(self.images_root, seq_folder, 'h36m_meta_wbbox.mat')))
            else:
                self.meta_data.append(loadmat(os.path.join(self.images_root, seq_folder, 'h36m_meta.mat')))
            for i in range(self.meta_data[-1]['img_width'].shape[0]):
                self.index_mapping[index] = (idx, i)
                index += 1
        self.length = index
        # print(self.meta_data[0])

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        seq_id, frame_id = self.index_mapping[index]
        pose3d_world = self.meta_data[seq_id]['pose3d_world'][frame_id]
        R = self.meta_data[seq_id]['R']
        T = self.meta_data[seq_id]['T']
        cxy = self.meta_data[seq_id]['c'][0]
        fxy = self.meta_data[seq_id]['f'][0]
        pose3d = pose_world_to_cam(pose3d_world, R, T)
        pose2d = pose_cam_to_pixel(pose3d, cxy, fxy)
        image = cv2.imread(os.path.join(self.images_root, self.seqs_folder[seq_id],
                                        '%s_%06d.jpg' % (self.seqs_folder[seq_id], frame_id + 1))) / 255
        if self.with_bbox:
            bbox = self.meta_data[seq_id]['bbox'][frame_id].astype(np.int)
        else:
            bbox = None
        if self.is_aug and random.random() < self.config['aug_ratio']:
            pose2d, pose3d, bbox, image = augment(pose2d, pose3d, bbox, image, self.config['augmentation'],
                                                  self.config['aug_params'])
        if self.with_bbox:
            ratio = [self.config['out_res'][0] / (bbox[2] - bbox[0]), self.config['out_res'][1] / (bbox[3] - bbox[1])]
            pose2d[:, 0] = (pose2d[:, 0] - bbox[0]) * ratio[0]
            pose2d[:, 1] = (pose2d[:, 1] - bbox[1]) * ratio[1]
            image = image[bbox[1]:bbox[3], bbox[0]:bbox[2], :]
            image = cv2.resize(image, self.config['in_res'])

        if self.config['normalize_3d_pose']:
            pose3d = normalize_3d_pose(pose3d)

        if self.heatmap:
            heatmap = self.generate_heatmap(pose2d)
            return pose2d, pose3d, bbox, image, heatmap

        return pose2d, pose3d, bbox, image


if __name__ == '__main__':
    a = Human36mDataset('/home/zhongyu/data/Human3.6m', protocols=[1])
    for i in range(100, 110):
        pose2d, pose3d, _, img = a[i]
        img = draw_skeleton(img, np.concatenate((pose2d, np.ones((17, 1))), axis=1), a.skeleton)
        cv2.imshow('test', img)
        cv2.waitKey()
