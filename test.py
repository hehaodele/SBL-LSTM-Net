import numpy as np
import scipy.io as sio

class Dict(dict):
    """
    Example:
    m = Dict({'first_name': 'Eduardo'}, last_name='Pool', age=24, sports=['Soccer'])
    """
    def __init__(self, *args, **kwargs):
        super(Dict, self).__init__(*args, **kwargs)
        for arg in args:
            if isinstance(arg, dict):
                for k, v in arg.iteritems():
                    self[k] = v

        if kwargs:
            for k, v in kwargs.iteritems():
                self[k] = v

    def __getattr__(self, attr):
        return self[attr]

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __setitem__(self, key, value):
        super(Dict, self).__setitem__(key, value)
        self.__dict__.update({key: value})

    def __delattr__(self, item):
        self.__delitem__(item)

    def __delitem__(self, key):
        super(Dict, self).__delitem__(key)
        del self.__dict__[key]


def visualize(x, pred, label):
    import matplotlib.pyplot as plt
    assert len(x) == len(pred)
    xs = np.arange(len(x))
    pred_top = np.argsort(pred)[::-1][:len(label)]
    pred = np.exp(pred) / np.sum(np.exp(pred), axis=0) # softmax
    plt.figure(0)
    plt.plot(xs, x, 'r')
    plt.plot(xs, pred, 'b')
    plt.scatter(label - 1, np.ones(label.shape) * 0.5, alpha = 0.5)
    plt.scatter(pred_top, pred[pred_top], alpha = 0.5)
    plt.show()
    # plt.savefig('vis.png')

result = sio.loadmat('./test_samples.mat')
pred = result["predict"]
label = result["label"]
X = result["X"]
for x, p, l, in zip(X, pred, label):
    visualize(x,p,l)

