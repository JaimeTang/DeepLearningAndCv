import cv2
import numpy as np

def stitch(imageA, imageB, ratio=0.75, reprojThresh=4.0,showMatches=False):

    kpsA, desA = detectAndDescribe(imageA)
    kpsB, desB = detectAndDescribe(imageB)

    M = matchKeyPoints(kpsA, kpsB, desA, desB, ratio, reprojThresh)

    if M is None:
        return None

    (matches, H, status) = M
    result = cv2.warpPerspective(imageA, H, (imageA.shape[1]+imageB.shape[1], imageA.shape[0]))

    result[0:imageB.shape[0], 0:imageB.shape[1]] = imageB

    if showMatches:
        vis = drawMatches(imageA, imageB, kpsA, kpsB, matches, status)
        return (result, vis)

    return result

def detectAndDescribe(image):

    sifi = cv2.xfeatures2d.SIFT_create()
    kps, des = sifi.detectAndCompute(image, None)

    kps = np.float32([kp.pt for kp in kps])

    return (kps, des)


def matchKeyPoints(kpsA, kpsB, desA, desB, ratio, reprojThresh):

    matcher = cv2.DescriptorMatcher_create("BruteForce")
    rawMatches = matcher.knnMatch(desA, desB, 2)
    matches = []

    for m in rawMatches:

        if len(m) == 2 and m[0].distance < m[1].distance * ratio:
            matches.append((m[0].trainIdx, m[0].queryIdx))

    if len(matches) > 4:
        # construct the two sets of points
        ptsA = np.float32([kpsA[i] for (_, i) in matches])
        ptsB = np.float32([kpsB[i] for (i, _) in matches])

        # compute the homography between the two sets of points
        (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, reprojThresh)

        # return the matches along with the homograpy matrix
        # and status of each matched point
        return (matches, H, status)
    return None

def drawMatches(imageA, imageB, kpsA, kpsB, matches, status):

    (hA, wA) = imageA.shape[:2]
    (hB, wB) = imageB.shape[:2]
    vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
    vis[0:hA, 0:wA] = imageA
    vis[0:hB, wA:] = imageB

    # loop over the matches
    for ((trainIdx, queryIdx), s) in zip(matches, status):
        # only process the match if the keypoint was successfully
        # matched
        if s == 1:
            # draw the match
            ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
            ptB = (int(kpsB[trainIdx][0]) + wA, int(kpsB[trainIdx][1]))
            cv2.line(vis, ptA, ptB, (0, 255, 0), 1)

    # return the visualization
    return vis

def main():
    img_B = cv2.imread("img/book_left.jpg")
    img_A= cv2.imread("img/book_right.jpg")
    img_A = cv2.resize(img_A, (int(img_A.shape[1]*0.5), int(img_A.shape[0]*0.5)), 0, 0)
    img_B = cv2.resize(img_B, (int(img_B.shape[1]*0.5), int(img_B.shape[0]*0.5)), 0, 0)
    output, vis = stitch(img_A, img_B, showMatches=True)
    cv2.imwrite("img/book_stitch.jpg", output)

    cv2.imshow("img_A", img_A)
    cv2.imshow("img_B", img_B)
    cv2.imshow("img_vis", vis)
    cv2.imshow("img_output", output)
    key = cv2.waitKey()
    if key == 27:
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()