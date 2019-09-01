import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import cv2


folder_list = ['I', 'II']
train_boarder = 112

def channel_norm(img):
    # img: ndarray, float32
    mean = np.mean(img)
    std = np.std(img)
    pixels = (img - mean) / (std + 0.0000001)
    return pixels

def parse_line(line):
    line_parts = line.strip().split()
    img_name = line_parts[0]
    rect = list(map(int, list(map(float, line_parts[1:5]))))
    landmarks = list(map(float, line_parts[5: len(line_parts)]))
    return img_name, rect, landmarks

class Normalize(object):
    """
        Resieze to train_boarder x train_boarder. Here we use 112 x 112
        Then do channel normalization: (image - mean) / std_variation
    """
    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        image_resize = np.asarray(
                            image.resize((train_boarder, train_boarder), Image.BILINEAR),
                            dtype=np.float32)       # Image.ANTIALIAS)
        image = channel_norm(image_resize)
        return {'image': image,
                'landmarks': landmarks
                }

class ToTensor(object):
    """
        Convert ndarrays in sample to Tensors.
        Tensors channel sequence: N x C x H x W
    """
    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        # image = image.transpose((2, 0, 1))
        image = np.expand_dims(image, axis=0)
        return {'image': torch.from_numpy(image),
                'landmarks': torch.from_numpy(landmarks)}

class FaceLandmarksDataset(Dataset):
    # Face Landmarks Dataset
    def __init__(self, src_lines, transform=None):
        '''
        :param src_lines: src_lines
        :param train: whether we are training or not
        :param transform: data transform
        '''
        self.lines = src_lines
        self.transform = transform

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        img_name, rect, landmarks = parse_line(self.lines[idx])
        # image
        img = Image.open(img_name).convert('L')
        img_crop = img.crop(tuple(rect))
        landmarks = np.array(landmarks).astype(np.float32)

		# you should let your landmarks fit to the train_boarder(112)
		# please complete your code under this blank
		# your code:
        landmarks = landmarks.reshape(-1, 2)

        top = rect[1]
        left = rect[0]
        landmarks  -= [left, top]

        sample = {'image': img_crop, 'landmarks': landmarks}
        sample = self.transform(sample)
        sample['rect'] = rect
        return sample

def load_data(phase):
    data_file = phase + '.txt'
    with open(data_file) as f:
        lines = f.readlines()

    if phase == 'Train' or phase == 'train':
        ##tsfm = transforms.Compose([Normalize(), ToTensor()])
        tsfm = transforms.Compose([ToTensor()])
    else:
        ##tsfm = transforms.Compose([Normalize(), ToTensor()])
        tsfm = transforms.Compose([ToTensor()])
    data_set = FaceLandmarksDataset(lines, transform=tsfm)
    return data_set

def get_train_test_set():
    train_set = load_data('train')
    valid_set = load_data('test')
    return train_set, valid_set

def main():
    train_set = load_data('test')

    for i in range(0, len(train_set)):
        sample = train_set[i]
        img = sample['image'][0].numpy()
        landmarks = sample['landmarks'].numpy()

        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        # rect = sample['rect']
        #cv2.rectangle(img, (rect[0], rect[1]), (rect[2], rect[3]), (0, 255, 0), 2)

        for point in  landmarks:
            cv2.circle(img, tuple(point), 1, color=(0, 0, 255), thickness=-1)

        cv2.imshow("img", img)
        key = cv2.waitKey()
        if key == 27:
            exit()
        cv2.destroyAllWindows()