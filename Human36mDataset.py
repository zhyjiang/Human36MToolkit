import os
import json
import glob
from scipy.io import loadmat
import cv2
import numpy as np

import torch
import torch.nn as nn

from utils.tools import pose_world_to_cam, pose_cam_to_pixel, draw_skeleton


class Human36mDataset(torch.utils.data.Dataset):
    def __init__(self, data_root, protocols=[1, 5, 6, 7, 8], is_train=True, is_aug=False):
        super(Human36mDataset, self).__init__()
        self.protocols = protocols
        self.is_train = is_train
        self.data_root = data_root
        self.is_aug = is_aug
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
                                        '%s_%06d.jpg' % (self.seqs_folder[seq_id], frame_id + 1)))
        return pose2d, pose3d, image


if __name__ == '__main__':
    a = Human36mDataset('/home/zhongyu/data/Human3.6m')
    for i in range(100, 110):
        pose2d, pose3d, img = a[i]
        img = draw_skeleton(img, np.concatenate((pose2d, np.ones((17, 1))), axis=1), a.skeleton)
        cv2.imshow('test', img)
        cv2.waitKey()
