## Logistic Regression
###############################
import numpy as np
np.random.seed(42)

def sigmoid(w, b, x):
    result = 1/(1+np.exp(w*x+b))
    return result

def eval_loss(w, b, x, y):
    avg_loss = np.mean(0.5 * np.square(sigmoid(w,x,b) - y))
    return avg_loss

def gradient(y_pred, y_true, x, w, b):
    loss = y_pred - y_true
    dw = -loss * sigmoid(w, b, x) * (1-sigmoid(w, b, x)) * x
    db = -loss * sigmoid(w, b, x) * (1-sigmoid(w, b, x))
    return dw, db

def cal_step_gradient(batch_x, batch_y, w, b, lr):
    y_pred = sigmoid(w, b ,batch_x)
    dw, db = gradient(y_pred, batch_y, batch_x, w, b)

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
    x_train = np.random.randint(-100, 100, num_samples)*np.random.random(num_samples)
    y_train = sigmoid(w, b, x_train)

    return x_train, y_train, w, b

def run():
    num_samples = 100
    lr = 1.5
    max_iter = 10000
    batch_size = 50

    x_train, y_train, w, b = gen_sample_data(num_samples)
    train(x_train, y_train, batch_size, lr, max_iter, num_samples)
    print("--"*20)
    print("Real w is {:.4f}, b is {:.4f}".format(w,b))

if __name__ == '__main__':
    run()