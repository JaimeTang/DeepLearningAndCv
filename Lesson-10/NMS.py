import numpy as np

def NMS(lists, thresh):
    # lists is a list. lists[0:4]: x1, y1, x2, y2; lists[4]: score
    dets = np.array(lists)
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    area = (x2 - x1 + 1)*(y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], x1[order[1:]])
        xx2 = np.maximum(x1[i], x1[order[1:]])
        yy2 = np.maximum(x1[i], x1[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w*h

        ovr = inter/(area[i]+area[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

        return keep


def main():
    lists = []
    thre = 0.5
    NMS(lists, thre)


if __name__=='__main__':
    main()