# SIFT
---

## 1. SIFT特征点和特征描述提取

SIFT算法广泛使用在计算机视觉领域，在OpenCV中也对其进行了实现。

```python
import cv2 
#这里使用的Python 3
def sift_kp(image):
    gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d_SIFT.create()
    kp,des = sift.detectAndCompute(image,None)
    kp_image = cv2.drawKeypoints(gray_image,kp,None)
    return kp_image,kp,des
```


## 2. SIFT特征点匹配

SIFT算法得到了图像中的特征点以及相应的特征描述，如何把两张图像中的特征点匹配起来呢？一般的可以使用K近邻（KNN）算法。K近邻算法求取在空间中距离最近的K个数据点，并将这些数据点归为一类。在进行特征点匹配时，一般使用KNN算法找到最近邻的两个数据点，如果最接近和次接近的比值大于一个既定的值，那么我们保留这个最接近的值，认为它和其匹配的点为good match（有Lowe在SIFT论文中提出）。

```python
import cv2
def get_good_match(des1,des2):
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)
    return good
```

## 3. 单应性矩阵 Homography Matrix

通过上面的步骤，我们找到了若干两张图中的匹配点，如何将其中一张图通过旋转、变换等方式将其与另一张图对齐呢？这就用到了单应性矩阵了。Homography这个词由Homo和graphy，Homo意为同一，graphy意为图像，也就是同一个东西产生的图像。不同视角的图像上的点具有如下关系：

![](img/dx-1.png)

其中[x1 y1 1]和[x2 y2 1]分别表示对应像素的齐次坐标。单应性矩阵就是下面这个矩阵：

![](img/dx-2.png)


可以看到，单应性矩阵有八个参数，如果要解这八个参数的话，需要八个方程，由于每一个对应的像素点可以产生2个方程(x一个，y一个)，那么总共只需要四个像素点就能解出这个单应性矩阵。

在Opencv-Python中可以使用如下方式：

```python
import cv2
H, status = cv2.findHomography(ptsA,ptsB,cv2.RANSAC,ransacReprojThreshold)
```

这里就出现了一个问题，我们用SIFT算法已经找到了若干个匹配的点了（几十个甚至上百个），那么应该选择哪四个点来计算Homography matrix呢？ RANSAC算法巧妙的解决了这一问题。

## 4. Random sample consensus:RANSAC（随机抽样一致算法）

说道估算模型的算法，很容易想到最小二乘法。最小二乘法在数据误差比较小的情况下是可以的，但是针对噪声很大的数据集的时候，容易出问题，我们可以通过下面的例子来说明

![](img/dx-3.png)

很明显，我们的数据应该是一条由左下到右上的一条线，但是由于离群数据太多（左上和右下区域），如果用最小二乘法的话，就无法准确的找到我们期望的模型（因为最小二乘法无法剔除误差很大的点，一视同仁的将所有的点都用于模型的计算）。**RANSAC（Random Sample Consensus）**算法的不同之处就在于，它能够有效的去除误差很大的点，并且这些点不计入模型的计算之中。该算法的框架如下：

- 在数据中随机的选择几个点设定为内群（也就是用来计算模型的点）
- 用这些选择的数据计算出一个模型
- 把其它刚才没选到的点带入刚才建立的模型中，计算是否为内群（也就是看这些点是否符合模型，如果符合模型的话，将新的点也计入内群）
- 如果此时的内群数量足够多的话，可以认为这个模型还算OK，那么就用现在的内群数据重新计算一个稍微好些的模型。
- 重复以上步骤，最后，我们保留那个内群数量最多的模型。


可以看到，RANSAC利用其独有的方式，在每一次的计算中都将一些数据点排除在外，起到了消除它们噪声的作用，并在此基础上计算模型。

所以，在得到了众多的匹配点以后，使用RANSAC算法，每次从中筛选四个随机的点，然后求得H矩阵，不断的迭代，直到求得最优的H矩阵为止。所以在前面使用cv2.findHomography方法时，参数cv2.RANSAC表明使用RANSAC算法来筛选关键点

```python
import cv2
H, status = cv2.findHomography(ptsA,ptsB,cv2.RANSAC,reprojThresh)
#其中H为求得的单应性矩阵矩阵
#status则返回一个列表来表征匹配成功的特征点。
#ptsA,ptsB为关键点
#cv2.RANSAC
#ransacReprojThreshold 则表示一对内群点所能容忍的最大投影误差
#Maximum allowed reprojection error to treat a point pair as an inlier (used in the RANSAC method only）
```

其中最大投影误差的计算公式如下：

![](img/dx-4.png)

## 5. 应用

在OpenCV提供了上述基础方法以后，我们可以利用这些方法来对图像进行匹配。

```python
def siftImageAlignment(img1,img2):
   _,kp1,des1 = sift_kp(img1)
   _,kp2,des2 = sift_kp(img2)
   goodMatch = get_good_match(des1,des2)
   if len(goodMatch) > 4:
       ptsA= np.float32([kp1[m.queryIdx].pt for m in goodMatch]).reshape(-1, 1, 2)
       ptsB = np.float32([kp2[m.trainIdx].pt for m in goodMatch]).reshape(-1, 1, 2)
       ransacReprojThreshold = 4
       H, status =cv2.findHomography(ptsA,ptsB,cv2.RANSAC,ransacReprojThreshold);
       imgOut = cv2.warpPerspective(img2, H, (img1.shape[1],img1.shape[0]),flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
   return imgOut,H,status
```

上面的函数首先找出img1和img2中的特征点，然后找到较好的匹配点对，最后通过warpPerspective方法对图像img2进行投影映射。其计算公式如下：

![](img/dx-5.png)

其中：

- 第一个参数为需要投影的图像（img2）
- 第二个参数为单应性矩阵（H）
- 第三个参数为所得图像的矩阵大小（（img1.shape[1],img1.shape[0]) ）
- 最后的参数cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP，为插值时使用的插值方法INTER_LINEAR，cv2.WARP_INVERSE_MAP则将M设置为dst--->src的方向变换。

我们简单的测试一下我们的函数：

```python
import numpy as np
import cv2
import Utility

img1 = cv2.imread('stack_alignment/IMG_4209.jpg')
img2 = cv2.imread('stack_alignment/IMG_4211.jpg')

result,_,_ = Utility.siftImageAlignment(img1,img2)
allImg = np.concatenate((img1,img2,result),axis=1)
cv2.namedWindow('Result',cv2.WINDOW_NORMAL)
cv2.imshow('Result',allImg)
cv2.waitKey(0)
```