import numpy as np
import random
import cv2

def distance_pp(point1, point2):
    return np.sqrt(np.sum((point2 - point1)**2))

def pose_world_to_cam(pose3d_world, R, T):
    pose3d = np.dot(R, (pose3d_world - T).T).T
    return pose3d


def pose_cam_to_pixel(pose3d, cxy, fxy):
    intrinsic = np.zeros((3, 3))
    intrinsic[0, 0] = fxy[0]
    intrinsic[1, 1] = fxy[1]
    intrinsic[0:2, 2] = cxy.T
    intrinsic[2, 2] = 1
    pose2d = intrinsic.dot(pose3d.T).T
    pose2d[:, :2] = pose2d[:, :2] / pose2d[:, 2:3]
    return pose2d[:, :2]


def draw_skeleton(img, pose2d, skeleton, thre=0.5):
    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    for i in range(pose2d.shape[0]):
        if pose2d[i, 2] > thre:
            cv2.circle(img, tuple(pose2d[i, :2].astype(np.int)), 2, color, -1)
    for i in range(len(skeleton)):
        if pose2d[skeleton[i][0], 2] > thre and pose2d[skeleton[i][1], 2] > thre:
            cv2.line(img, tuple(pose2d[skeleton[i][0], :2].astype(np.int)),
                     tuple(pose2d[skeleton[i][1], :2].astype(np.int)), color, 1)
    return img


def normalize_3d_pose(pose3d):
    pose_center = (pose3d[1] + pose3d[4]) / 2
    dist = distance_pp(pose_center, pose3d[8])
    pose3d = pose3d - pose_center
    pose3d = pose3d / dist
    return pose3d


def get_keypoints(heatmap):
    keypoints = []
    for i in range(heatmap.shape[0]):
        index = np.argmax(heatmap[i, :, :])
        keypoints.append([int(index % heatmap.shape[2]), int(index // heatmap.shape[2]), np.max(heatmap[i, :, :])])
    return np.array(keypoints)

