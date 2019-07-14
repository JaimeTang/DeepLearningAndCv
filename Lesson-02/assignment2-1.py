import cv2
import numpy as np

def medianBlur(img, kernel, padding_way):
    """
    img & kernel is List of List; padding_way a string
    Please finish your code under this blank
    REPLICA & ZERO
    """
    kernel = np.flipud(np.fliplr(kernel))
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

    kernel_size = kernel.shape[0]
    # step=1,kernel_size=3
    for y in range(img.shape[1]):
        for x in range(img.shape[0]):
            output[x, y] = (kernel*img_padding[x:x+kernel_size, y:y+kernel_size]).sum()
    return output

def main():
    img = cv2.imread("img/img.jpg", 0)
    padding_way = input("Image Padding Way (Input Number.):\n 1. REPLICA \n 2. ZERO \n --- \n")
    kernel = np.array(
        [
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]
        ]
    )
    output = medianBlur(img, kernel, padding_way)

    cv2.imshow("or_img", img)
    cv2.imshow("filter_img", output)
    key = cv2.waitKey()
    if key == 27:
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()