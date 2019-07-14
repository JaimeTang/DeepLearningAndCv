import cv2
import numpy as np

def ransacMatching(A, B):
    # A & B: List of List
    img_r = B
    img_l = A
    sift = cv2.xfeatures2d.SIFT_create()
    kp_r, des_r = sift.detectAndCompute(img_r, None)
    kp_l, des_l = sift.detectAndCompute(img_l, None)

    index_params = dict(algorithm=0, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des_r, des_l, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    if len(good) > 4:
        src_pts = np.float32([kp_r[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_l[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()

        h, w = img_r.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)

        img_l = cv2.polylines(img_l, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

        draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                           singlePointColor=None,
                           matchesMask=matchesMask,  # draw only inliers
                           flags=2)

        output = cv2.drawMatches(img_r, kp_r, img_l, kp_l, good, None, **draw_params)

        return output

    else:
        print("Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT))
        matchesMask = None

def main():
    img_right = cv2.imread("img/book_right.jpg", 0)
    img_left = cv2.imread("img/book_left.jpg", 0)
    output = ransacMatching(img_right,img_left)
    cv2.imwrite("img/book_output.jpg", output)

    cv2.imshow("img_right", img_right)
    cv2.imshow("img_left", img_left)
    cv2.imshow("img_output", output)
    key = cv2.waitKey()
    if key == 27:
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()