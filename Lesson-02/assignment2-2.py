import cv2
import numpy as np

def ransacMatching(A, B):
    # A & B: List of List
    H, mask = cv2.findHomography(A, B, cv2.RANSAC, 4.0)



def main():
    img = cv2.imread("img/img.jpg", 0)


    cv2.imshow("or_img", img)
    cv2.imshow("filter_img", output)
    key = cv2.waitKey()
    if key == 27:
        cv2.destroyAllWindows()



if __name__ == '__main__':
    main()