import numpy as np
import cv2


class BSF_SCRG(object):
    def __init__(self, use_centroid=False):
        self.use_centroid = use_centroid
        self.reset()

    def update(self, pred, label, in_img):
        label = label > 0

        # analysis target number
        num_labels, labels, _, centroids = cv2.connectedComponentsWithStats(label.astype(np.uint8))
        assert num_labels > 1

        # pick up background region
        mask_back = label == 0
        patch_back_in = in_img[mask_back]
        patch_back_out = pred[mask_back]

        mean_back_in = np.mean(patch_back_in)
        mean_back_out = np.mean(patch_back_out)

        # pick up target region separately
        for i in range(1, num_labels-1):
            centroid = centroids[i]
            mask_target = (labels == i)

            patch_target_in = in_img[mask_target]
            patch_target_out = pred[mask_target]

            if self.use_centroid:
                # use centroid pixel value to replace mean of target region
                assert mask_target[centroid] == i
                mean_target_in = in_img[centroid[0], centroid[1]]
                mean_target_out = pred[centroid[0], centroid[1]]
            else:
                mean_target_in = np.mean(patch_target_in)
                mean_target_out = np.mean(patch_target_out)

            # update parameters for every target
            self.sin = np.append(self.sin, np.abs(mean_target_in - mean_back_in))
            self.sout = np.append(self.sout, np.abs(mean_target_out - mean_back_out))
            self.cin = np.append(self.cin, np.std(patch_back_in))
            self.cout = np.append(self.cout, np.std(patch_back_out))

    def get(self):
        scrgs = (self.sout / self.cout) / (self.sin / self.cin)
        bsfs = self.cin / self.cout
        return np.mean(scrgs), np.mean(bsfs)

    def get_all(self):
        return self.sin, self.sout, self.cin, self.cout

    def reset(self):
        self.sin = np.array([])
        self.sout = np.array([])
        self.cin = np.array([])
        self.cout = np.array([])