# from Human36MToolkit.utils.tools import distance_pp, distance_ll_and_closest_points
# import numpy as np
#
#
# class BodyCylinder:
#     cylinder_limb = ((1, 2), (2, 3), (4, 5), (5, 6), (11, 12), (12, 13), (14, 15), (15, 16))
#     cylinder_head = (8, 10)
#     cylinder_torso = (8, (1, 4))
#     n_joints = 17
#
#     def __init__(self, head_radius=100, limb_radius=50):
#         self.limb_radius = limb_radius
#         self.head_radius = head_radius
#
#     def __call__(self, pose3d):
#         torso_radius = distance_pp(pose3d[11], pose3d[14]) / 2
#         visible_list = [1 for i in range(self.n_joints)]
#         for i in range(self.n_joints):
#             for limb in self.cylinder_limb:
#                 if i not in limb:
#                     dist, p1, p2 = distance_ll_and_closest_points([np.array([0, 0, 0]), pose3d[i]],
#                                                                   [pose3d[limb[1]], pose3d[limb[0]] - pose3d[limb[1]]])
#                     if dist < self.limb_radius:
#                         if p1 is None and (np.sum(pose3d[i] ** 2) > np.sum(pose3d[limb[1]] ** 2) or
#                                            np.sum(pose3d[i] ** 2) > np.sum(pose3d[limb[0]] ** 2)):
#                             visible_list[i] = 0
#                             continue
#                         elif np.sum(pose3d[i] ** 2) > np.sum(p1 ** 2) and \
#                                 distance_pp(p2, pose3d[limb[1]]) + distance_pp(p2, pose3d[limb[0]]) \
#                                 - distance_pp(pose3d[limb[0]], pose3d[limb[1]]) < 1e-4:
#                             visible_list[i] = 0
#                             continue
#             if visible_list[i] == 0:
#                 continue
#             if i not in self.cylinder_head:
#                 dist, p1, p2 = distance_ll_and_closest_points([np.array([0, 0, 0]), pose3d[i]],
#                                                               [pose3d[self.cylinder_head[1]],
#                                                                pose3d[self.cylinder_head[0]] - pose3d[
#                                                                    self.cylinder_head[1]]])
#                 if dist < self.head_radius:
#                     if p1 is None and (np.sum(pose3d[i] ** 2) > np.sum(pose3d[self.cylinder_head[1]] ** 2) or
#                                        np.sum(pose3d[i] ** 2) > np.sum(pose3d[self.cylinder_head[0]] ** 2)):
#                         visible_list[i] = 0
#                         continue
#                     elif np.sum(pose3d[i] ** 2) > np.sum(p1 ** 2) and \
#                             distance_pp(p2, pose3d[self.cylinder_head[1]]) \
#                             + distance_pp(p2, pose3d[self.cylinder_head[0]]) \
#                             - distance_pp(pose3d[self.cylinder_head[0]], pose3d[self.cylinder_head[1]]) < 1e-4:
#                         visible_list[i] = 0
#                         continue
#             if i not in [0, 1, 4, 7, 8, 11, 14]:
#                 torso0 = pose3d[self.cylinder_torso[0]]
#                 torso1 = (pose3d[self.cylinder_torso[1][0]] + pose3d[self.cylinder_torso[1][1]]) / 2
#                 dist, p1, p2 = distance_ll_and_closest_points([np.array([0, 0, 0]), pose3d[i]],
#                                                               [torso0, torso1 - torso0])
#                 if dist < torso_radius:
#                     if p1 is None and (np.sum(pose3d[i] ** 2) > np.sum(torso0 ** 2) or
#                                        np.sum(pose3d[i] ** 2) > np.sum(torso1 ** 2)):
#                         visible_list[i] = 0
#                         continue
#                     elif np.sum(pose3d[i] ** 2) > np.sum(p1 ** 2) and distance_pp(p2, torso0) + \
#                             distance_pp(p2, torso1) - distance_pp(torso0, torso1) < 1e-4:
#                         visible_list[i] = 0
#                         continue
#
#         return visible_list
