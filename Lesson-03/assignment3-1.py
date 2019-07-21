## Linear Regression
###############################
import numpy as np
np.random.seed(42)

def predict(w, b, x):
    y_pred = w * x + b
    return y_pred

def eval_loss(w, b, x, y):
    avg_loss = np.mean(0.5 * np.square(w * x + b - y))
    return avg_loss

def gradient(y_pred, y_true, x):
    loss = y_pred - y_true
    dw = loss * x
    db = loss
    return dw, db

def cal_step_gradient(batch_x, batch_y, w, b, lr):
    y_pred = predict(w, b, batch_x)
    dw, db = gradient(y_pred, batch_y, batch_x)

    avg_dw = np.mean(dw)
    avg_db = np.mean(db)
    w -= lr * avg_dw
    b -= lr * avg_db

    return w, b

def train(x_train, y_train, batch_size, lr, max_iter, num_samples):
    w = 0
    b = 0

    for i in range(max_iter):
        batch_idxs = np.random.choice(num_samples, batch_size)
        batch_x, batch_y = x_train[batch_idxs], y_train[batch_idxs]
        w, b = cal_step_gradient(batch_x, batch_y, w, b, lr)

        print('w:{:.4f}, b:{:.4f}'.format(w, b))
        print('loss is {:.4f}'.format(eval_loss(w, b, x_train, y_train)))

def gen_sample_data(num_samples=100):
    w = np.random.randint(0, 10)+np.random.random()
    b = np.random.randint(0, 5)+np.random.random()
    x_train = np.random.randint(0, 100, num_samples)*np.random.random(num_samples)
    y_train = w * x_train + b + np.random.random(num_samples) * np.random.randint(-1, 1,num_samples)

    return x_train, y_train, w, b

def run():
    num_samples = 100
    lr = 0.001
    max_iter = 10000
    batch_size = 50

    x_train, y_train, w, b = gen_sample_data(num_samples)
    train(x_train, y_train, batch_size, lr, max_iter, num_samples)
    print("--"*20)
    print("Real w is {}, b is {}".format(w,b))

if __name__ == '__main__':
    run()