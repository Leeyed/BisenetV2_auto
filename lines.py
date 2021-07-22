import cv2 as cv
import numpy as np
from skimage import morphology
import configparser
import os

# configpath = '/opt/config/ai-product-injection-mold-inserts/detectcontroll/'
TASK_PY_PATH = os.path.split(os.path.realpath(__file__))[0]
configpath = TASK_PY_PATH


def emphasize(img, cov_w=7, cov_h=7, factor=1.0):
    if len(img.shape) == 3:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = img.astype(float)
    cov = np.ones([cov_w, cov_h], dtype=float) / (cov_w * cov_h)
    mean = cv.filter2D(img, -1, cov).astype(float)
    img = img + (img - mean) * factor
    img[np.where(img <= 0)] = 0
    img[np.where(img >= 255)] = 255
    img = img.astype(np.uint8)
    return img


def get_skeleton_points(skeletonmap, angle_thre1=30, angle_thre2=45):
    contours, _ = cv.findContours(skeletonmap, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours.sort(key=cnt_area1, reverse=True)
    points = cv.approxPolyDP(contours[0], 1, False)
    points_ori = np.squeeze(points, 1)
    points = points_ori.copy()
    try:
        if len(points) > 1:
            angles = []
            lengths = []
            lengths1 = []
            end_indexs = []
            # 计算各点间的距离与直线夹角
            for i in range(len(points) - 1):
                x1, y1 = points[i]
                x2, y2 = points[i + 1]
                if x2 - x1 == 0 and y2 > y1:
                    angles.append(90)
                elif x2 - x1 == 0 and y2 < y1:
                    angles.append(270)
                else:
                    k = (y2 - y1) / (x2 - x1)
                    if x1 - x2 > 0:
                        angles.append(np.arctan(k) * 57.29577 + 180)
                    elif y1 - y2 > 0:
                        angles.append(np.arctan(k) * 57.29577 + 360)
                    else:
                        angles.append(np.arctan(k) * 57.29577)
                lengths.append(np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))
            # print(len(points_ori))
            # print('angles', angles)
            # print('lengths', lengths)
            # 计算以各点为起始点，符合angle_thre1和angle_thre2的线段长度
            for i in range(len(points) - 1):
                length = lengths[i].copy()
                end_index = i + 1

                if i < len(points) - 2:
                    for j in range(i + 1, len(points) - 1):
                        in_angle1 = np.abs(angles[j - 1] - angles[j])
                        in_angle2 = np.abs(angles[i] - angles[j])
                        if (in_angle1 if in_angle1 <= 180 else 360 - in_angle1) > angle_thre1 or (
                                in_angle2 if in_angle2 <= 180 else 360 - in_angle2) > angle_thre2:
                            break
                        length += lengths[j]
                        end_index = j + 1
                    lengths1.append(length)
                    end_indexs.append(end_index)

            # print('lengths1', lengths1)
            # print('end_indexs', end_indexs)
            start_index = lengths1.index(max(lengths1))
            end_index = end_indexs[start_index]
            # print(start_index, end_index)
            return points_ori, points[start_index:end_index + 1]
        else:
            return [], []
    except:
        return [], []


def get_line_region(contours, minArea, maxArea):
    contourlist = []
    for contour in contours:
        area = cv.contourArea(contour)
        if area > minArea and area < maxArea:
            contourlist.append(contour)
    return contourlist


def get_max_Contour(Contours):
    arealist = []
    for Contour in Contours:
        arealist.append(cv.contourArea(Contour))
    return arealist.index(max(arealist))


def cnt_len(contour):
    try:
        xmin = np.min(contour[:, 0, 0])
        ymin = np.min(contour[:, 0, 1])
        xmax = np.max(contour[:, 0, 0])
        ymax = np.max(contour[:, 0, 1])
        return np.sqrt(np.power(xmax - xmin, 2) + np.power(ymax - ymin, 2))
    except:
        return 0


def cnt_area(contour):
    return cv.contourArea(contour)


def cnt_area1(contour):
    return contour.shape[0]


def get_keypoints(skeletonmap):
    values = cv.filter2D(skeletonmap, -1, np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]]))
    values = cv.multiply(values, skeletonmap)
    y1, x1 = np.where(values == 1)
    y3, x3 = np.where(values >= 3)

    extreme_points = np.vstack([x1, y1]).T.tolist()
    crossover_points = np.vstack([x3, y3]).T.tolist()
    return extreme_points, crossover_points


class get_lines():
    def __init__(self):
        conf = configparser.ConfigParser()
        conf.read(os.path.join(configpath, 'lines_pars.cfg'))
        self.zoom1 = float(conf.get("Lines_Config", "zoom1"))
        self.zoom2 = float(conf.get("Lines_Config", "zoom2"))
        self.emphasize_cov = int(conf.get("Lines_Config", "emphasize_cov"))
        self.emphasize_factor = float(conf.get("Lines_Config", "emphasize_factor"))
        self.canny_thre1 = int(conf.get("Lines_Config", "canny_thre1"))
        self.canny_thre2 = int(conf.get("Lines_Config", "canny_thre2"))
        self.skeleton_angle_thre1 = int(conf.get("Lines_Config", "skeleton_angle_thre1"))
        self.skeleton_angle_thre2 = int(conf.get("Lines_Config", "skeleton_angle_thre2"))
        self.mask_thicknss = int(conf.get("Lines_Config", "mask_thicknss"))
    def get_lines_mask(self, imgori, mask):
        try:
            img = cv.cvtColor(imgori, cv.COLOR_BGR2GRAY)
            contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            if len(contours) > 1:
                contours.sort(key=cnt_area, reverse=True)
            x0 = np.min(contours[0][:, 0, 0])
            y0 = np.min(contours[0][:, 0, 1])
            x1 = np.max(contours[0][:, 0, 0])
            y1 = np.max(contours[0][:, 0, 1])

            img = img[y0:y1, x0:x1]
            mask = mask[y0:y1, x0:x1]

            h, w = img.shape[:2]
            img = cv.resize(img, (int(self.zoom1 * w), int(self.zoom1 * h)), cv.INTER_NEAREST)
            mask = cv.resize(mask, (int(self.zoom1 * w), int(self.zoom1 * h)), cv.INTER_NEAREST)

            kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (15, 15))
            mask = cv.erode(mask, kernel)
            img = emphasize(img, self.emphasize_cov, self.emphasize_cov, self.emphasize_factor)
            # Canny
            binary = cv.Canny(img, self.canny_thre1, self.canny_thre2)
            # binary = cv.Canny(img, 40, 80)
            binary = cv.multiply(mask, binary)
            # cv.waitKey(0)
            kernel2 = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
            # binary = cv.morphologyEx(binary, cv.MORPH_DILATE, kernel2)
            binary = cv.morphologyEx(binary, cv.MORPH_CLOSE, kernel2)

            contours, _ = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            contours.sort(key=cnt_len, reverse=True)
            darkmap = np.zeros(img.shape, dtype=np.uint8)
            cv.drawContours(darkmap, contours, 0, (255, 255, 255), -1)
            darkmap = cv.resize(darkmap, (int(self.zoom2 * w), int(self.zoom2 * h)), cv.INTER_NEAREST)
            kernel3 = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
            darkmap = cv.morphologyEx(darkmap, cv.MORPH_CLOSE, kernel3)
            _, darkmap = cv.threshold(darkmap, 128, 255, cv.THRESH_BINARY)

            darkmap[darkmap == 255] = 1
            skeleton0 = morphology.skeletonize(darkmap).astype(np.uint8)
            points_ori, points = get_skeleton_points(skeleton0, self.skeleton_angle_thre1, self.skeleton_angle_thre2)
            darkmap = np.zeros(imgori.shape, dtype=np.uint8)

            if len(points) > 0:
                points[:, 0] = points[:, 0] / self.zoom2 + x0
                points[:, 1] = points[:, 1] / self.zoom2 + y0
                cv.polylines(darkmap, [points], False, (1, 1, 1), self.mask_thicknss)
            return darkmap[:, :, 0]
        except:
            return np.zeros(imgori.shape, dtype=np.uint8)[:, :, 0]

    def get_lines_boxs(self, imgori, mask, boxs_num=1, box_size=128):
        def takesecond(S):
            return S[1]

        def takefirst(S):
            return S[0]

        try:
            if len(mask.shape) == 3:
                mask = mask[:, :, 0]
            boxs = []
            mask = self.get_lines_mask(imgori, mask)
            contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            if len(contours) > 0:
                contour = contours[0]
                xmin = np.min(contour[:, :, 0])
                ymin = np.min(contour[:, :, 1])
                xmax = np.max(contour[:, :, 0])
                ymax = np.max(contour[:, :, 1])
                points = contour[:, 0, :].tolist()
                if ymax - ymin > xmax - xmin:
                    points.sort(key=takesecond)
                else:
                    points.sort(key=takefirst)

                length = len(points)
                for i in range(1, 1 + boxs_num):
                    point = points[int(i * length / (boxs_num + 1))]
                    boxs.append([int(point[0] - box_size / 2), int(point[1] - box_size / 2),
                                 int(point[0] - box_size / 2) + int(box_size),
                                 int(point[1] - box_size / 2) + int(box_size)])
            return boxs
        except Exception as e:
            print(str(e))
            return []

GL=get_lines()
