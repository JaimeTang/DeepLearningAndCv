import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import cv2

#np.random.seed(42)


def assignment(df, center, colmap):
    
    for i in center.keys():
        df["distance_from_{}".format(i)] = np.sqrt((df["x"]-center[i][0])**2+(df["y"]-center[i][1])**2)

    distance_from_centroid_id = ['distance_from_{}'.format(i) for i in center.keys()]
    df["closest"] = df.loc[:, distance_from_centroid_id].idxmin(axis=1)
    df["closest"] = df["closest"].map(lambda x: int(x.lstrip('distance_from_')))
    df["color"] = df['closest'].map(lambda x: colmap[x])
    return df

def update(df, centroids):

    for i in centroids.keys():
        centroids[i][0] = np.mean(df[df['closest'] == i]['x'])
        centroids[i][1] = np.mean(df[df['closest'] == i]['y'])
    return centroids

def select_center(first_center, df, k):
    center = {}
    center[0]=first_center
    D_x_2 = (df["x"]-first_center[0])**2+(df["y"]-first_center[1])**2
    P_x = D_x_2/(D_x_2.sum())
    sum_x = P_x.cumsum()
    for i in range(1,k):
        next_center = random.random()
        for index,j in enumerate(sum_x):
            if j>next_center:
                break
        center[i]=list(df.iloc[index].values)
    return center


def main():
    df = pd.DataFrame({
        'x': [12, 20, 28, 18, 10, 29, 33, 24, 45, 45, 52, 51, 52, 55, 53, 55, 61, 64, 69, 72, 23],
        'y': [39, 36, 30, 52, 54, 20, 46, 55, 59, 63, 70, 66, 63, 58, 23, 14, 8, 19, 7, 24, 77]
    })

    k = 3

    first_center = [np.random.randint(0,80), np.random.randint(0,80)]
    center = select_center(first_center, df, k)

    colmap = {0:"r", 1:"g", 2:"b"}
    df = assignment(df, center, colmap)
    plt.scatter(df["x"], df["y"], c=df["color"],alpha=0.5,edgecolors='b')

    for i in center.keys():
        plt.scatter(*center[i], c=colmap[i], linewidths=6)

    plt.xlim(0,80)
    plt.ylim(0,80)
    plt.savefig("./img/img_0.png",format="png",dpi=300)
    plt.show()


    for i in range(10):

        key = cv2.waitKey()
        plt.close()

        closest_center = df['closest'].copy(deep=True)
        center = update(df, center)

        plt.scatter(df['x'], df['y'], color=df['color'], alpha=0.5, edgecolor='b')

        for j in center.keys():
            plt.scatter(*center[j], color=colmap[j], linewidths=6)
        plt.xlim(0, 80)
        plt.ylim(0, 80)
        plt.savefig("./img/img_{}.png".format(i+1),format="png",dpi=300)
        plt.show()

        df = assignment(df, center, colmap)

        if closest_center.equals(df['closest']):
            break


if __name__=='__main__':
    main()
