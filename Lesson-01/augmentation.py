"""
combine image crop, color shift, rotation and perspective transform together 

"""

import cv2
import random
import numpy as np

def random_crop(img):
    print("Random Crop.")
    height, width, _ = img.shape
    random_height = random.randint(0,height)
    random_width = random.randint(0,width)
    img_crop = img[random_height:height,random_width:width]
    return img_crop

def color_shift(img):
    print("Color Shift.")
    B, G, R = cv2.split(img)

    for i in (B,G,R):
        rand = random.randint(-50,50)
        if rand == 0:
            pass
        elif rand > 0:
            lim = 255-rand
            i[i > lim] = 255
            i[i <= lim] = (rand + i[i <= lim]).astype(img.dtype)
        elif rand < 0:
            lim = 0-rand
            i[i < lim] = 0
            i[i >= lim] = (rand + i[i >= lim]).astype(img.dtype)

    return img

def rotation_img(img, angle=30,scale=1):
    print("Rotation Image.")
    M = cv2.getRotationMatrix2D((img.shape[1]/2, img.shape[0]/2), angle, scale)
    img_rotation = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
    return img_rotation


def random_warp(img):
    print("Random Warp.")
    height, width, channels = img.shape
    # warp:
    random_margin = 60
    x1 = random.randint(-random_margin, random_margin)
    y1 = random.randint(-random_margin, random_margin)
    x2 = random.randint(width - random_margin - 1, width - 1)
    y2 = random.randint(-random_margin, random_margin)
    x3 = random.randint(width - random_margin - 1, width - 1)
    y3 = random.randint(height - random_margin - 1, height - 1)
    x4 = random.randint(-random_margin, random_margin)
    y4 = random.randint(height - random_margin - 1, height - 1)

    dx1 = random.randint(-random_margin, random_margin)
    dy1 = random.randint(-random_margin, random_margin)
    dx2 = random.randint(width - random_margin - 1, width - 1)
    dy2 = random.randint(-random_margin, random_margin)
    dx3 = random.randint(width - random_margin - 1, width - 1)
    dy3 = random.randint(height - random_margin - 1, height - 1)
    dx4 = random.randint(-random_margin, random_margin)
    dy4 = random.randint(height - random_margin - 1, height - 1)

    pts1 = np.float32([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
    pts2 = np.float32([[dx1, dy1], [dx2, dy2], [dx3, dy3], [dx4, dy4]])
    
    
    M_warp = cv2.getPerspectiveTransform(pts1, pts2)
    img_warp = cv2.warpPerspective(img, M_warp, (width, height))
    return img_warp

if __name__ == "__main__":
    img = cv2.imread("0.jpg")
    n = input("图像增强的数量：")
    transforms = [random_crop,color_shift,rotation_img,random_warp]
    for i in range(int(n)):
        do = random.choice(transforms)
        aug_img = do(img)
        cv2.imwrite("img_data/{}.jpg".format(i),aug_img)
