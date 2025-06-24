import numpy as np
import cv2
from skimage import measure

class PD_FA():
    def __init__(self,):
        super(PD_FA, self).__init__()
        self.image_area_total = []
        self.image_area_match = []
        self.dismatch_pixel = 0
        self.all_pixel = 0
        self.PD = 0
        self.target= 0
    def update(self, preds, labels, size):
        preds = preds / np.max(preds)  # normalize output to 0-1
        predits  = np.array(preds> 0.5).astype('int64')
        labelss = np.array(labels).astype('int64')

        # predits = preds> 0.5
        # labelss = labels

        image = measure.label(predits, connectivity=2)  # 标记8连通区域
        coord_image = measure.regionprops(image)        # 对不同连通区域进行操作
        label = measure.label(labelss , connectivity=2)
        coord_label = measure.regionprops(label)

        self.target    += len(coord_label)
        self.image_area_total = []
        self.image_area_match = []
        self.distance_match   = []
        self.dismatch         = []

        for K in range(len(coord_image)):
            area_image = np.array(coord_image[K].area)   # 预测图的  不同连通区域内的像素点数  2  area_image
            self.image_area_total.append(area_image)     # 预测图的  不同连通区域内的像素点数序列  2  image_area_total

        for i in range(len(coord_label)):
            centroid_label = np.array(list(coord_label[i].centroid))   # 掩膜图的  不同连通区域内的像素质心坐标  2  centroid_label
            for m in range(len(coord_image)):
                centroid_image = np.array(list(coord_image[m].centroid))   # 预测图的  不同连通区域内的像素质心坐标  2  centroid_image
                distance = np.linalg.norm(centroid_image - centroid_label)    # 二者质心的距离  2  distance
                area_image = np.array(coord_image[m].area)
                if distance < 3:
                    self.distance_match.append(distance)   # 距离小于3的distance  2  distance_match序列
                    self.image_area_match.append(area_image)   # 距离小于3的预测图像素点数  2  image_area_match序列

                    del coord_image[m]
                    break

        self.dismatch = [x for x in self.image_area_total if x not in self.image_area_match]   #
        self.dismatch_pixel +=np.sum(self.dismatch)
        self.all_pixel +=size[0]*size[1]
        self.PD +=len(self.distance_match)

    def get(self):
        Final_FA =  self.dismatch_pixel / self.all_pixel
        Final_PD =  self.PD /self.target
        return Final_PD, float(Final_FA)
