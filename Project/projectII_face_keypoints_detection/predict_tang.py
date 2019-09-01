from __future__ import print_function
import argparse
import torch
from torch.utils.data.sampler import SubsetRandomSampler
import os
from detector_tang import Net
import cv2
from data_tang import get_train_test_set

# 此部分代码针对stage 1中的predict。 是其配套参考代码
# 对于stage3， 唯一的不同在于，需要接收除了pts以外，还有：label与分类loss。

def predict(args, trained_model, model, valid_loader):
    model.load_state_dict(torch.load(os.path.join(args.save_directory, trained_model)))   # , strict=False
    model.eval()  # prep model for evaluation
    with torch.no_grad():
        for i, batch in enumerate(valid_loader):
            # forward pass: compute predicted outputs by passing inputs to the model
            img = batch['image']
            landmark = batch['landmarks']
            # generated
            output_pts = model(img)
            outputs = output_pts.numpy()[0]
            print('outputs: ', outputs)
            x = list(map(int, outputs[0: len(outputs): 2]))
            y = list(map(int, outputs[1: len(outputs): 2]))
            landmarks_generated = list(zip(x, y))
            # truth
            landmark = landmark.numpy()[0]
            x = list(map(int, landmark[0: len(landmark): 2]))
            y = list(map(int, landmark[1: len(landmark): 2]))
            landmarks_truth = list(zip(x, y))

            img = img.numpy()[0].transpose(1, 2, 0)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            for landmark_truth, landmark_generated in zip(landmarks_truth, landmarks_generated):
                cv2.circle(img, tuple(landmark_truth), 2, (0, 0, 255), -1)
                cv2.circle(img, tuple(landmark_generated), 2, (0, 255, 0), -1)
            out = img*255
            cv2.imshow(str(i), img)
            cv2.imwrite("img/img_{}.png".format(i), out)
            key = cv2.waitKey()
            if key == 27:
                exit()
            cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='Detector')
    parser.add_argument('--save-directory', type=str, default='trained_models',
                        help='learnt models are saving here')
    args = parser.parse_args()

    _, valid_set = get_train_test_set()
    valid_loader = torch.utils.data.DataLoader(dataset=valid_set, batch_size=1)

    trained_model = 'detector_epoch_0.pt'
    model = Net()

    predict(args, trained_model, model, valid_loader)

if __name__=='__main__':
    main()