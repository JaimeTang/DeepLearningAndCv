import cv2
import numpy as np

def ransacMatching(A, B):
    # A & B: List of List


def main():
    img = cv2.imread("img/img.jpg", 0)
    padding_way = input("Image Padding Way (Input Number.):\n 1. REPLICA \n 2. ZERO \n --- \n")
    """
        if padding_way !="1" or padding_way !="2":
        print("Wrong padding way.")
        return
    """
    kernel = (1/9)*np.array(
        [
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1]
        ]
    )
    print(kernel)
    output = medianBlur(img, kernel, padding_way)

    cv2.imshow("or_img", img)
    cv2.imshow("filter_img", output)
    key = cv2.waitKey()
    if key == 27:
        cv2.destroyAllWindows()



if __name__ == '__main__':
    main()