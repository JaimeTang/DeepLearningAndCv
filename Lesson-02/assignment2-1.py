import cv2
import numpy as np

def medianBlur(img, kernel, padding_way):
    """
    img & kernel is List of List; padding_way a string
    Please finish your code under this blank
    REPLICA & ZERO
    """
    kernel_size = kernel
    output = np.zeros_like(img)

    if padding_way == "1":
        img_padding = np.zeros((img.shape[0] + 2, img.shape[1] + 2))
        img_padding[1:-1, 1:-1] = img
        img_padding[0, 1:-1] = img[0, :]
        img_padding[-1, 1:-1] = img[-1, :]
        img_padding[1:-1, 0] = img[:, 0]
        img_padding[1:-1, -1] = img[:, -1]

    elif padding_way == "2":
        img_padding = np.zeros((img.shape[0] + 2, img.shape[1] + 2))
        img_padding[1:-1, 1:-1] = img

    for y in range(img.shape[1]):
        for x in range(img.shape[0]):
            temp = []
            temp = img_padding[x:x+kernel_size, y:y+kernel_size]
            output[x, y] = np.median(temp)
    return output

def main():
    img = cv2.imread("img/img.jpg", 0)
    padding_way = input("Image Padding Way (Input Number.):\n 1. REPLICA \n 2. ZERO \n --- \n")

    kernel_size = 4
    output = medianBlur(img, kernel_size, padding_way)
    cv2.imshow("original_img", img)
    cv2.imshow("filter_img", output)
    key = cv2.waitKey()
    if key == 27:
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()