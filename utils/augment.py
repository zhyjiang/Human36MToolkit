import cv2


class Augmentation:
    flip_pair = ((1, 4), (2, 5), (3, 6), (14, 11), (15, 12), (16, 13))

    def flip(self, pose2d, pose3d, bbox, image):
        image = cv2.flip(image, 1)
        h, w, _ = image.shape

        for pair in self.flip_pair:
            pose2d[pair[0]], pose2d[pair[1]] = pose2d[pair[1]].copy(), pose2d[pair[0]].copy()
            pose3d[pair[0]], pose3d[pair[1]] = pose3d[pair[1]].copy(), pose3d[pair[0]].copy()

        pose2d[:, 0] = w - pose2d[:, 0]
        pose3d[:, 0] = -pose3d[:, 0]

        if bbox is not None:
            bbox[0], bbox[2] = w - bbox[2], w - bbox[0]

        return pose2d, pose3d, bbox, image


def augment(pose2d, pose3d, bbox, image, augment_list):
    aug = Augmentation()
    for augment_method in augment_list:
        pose2d, pose3d, bbox, image = getattr(aug, augment_method)(pose2d, pose3d, bbox, image)

    return pose2d, pose3d, bbox, image
