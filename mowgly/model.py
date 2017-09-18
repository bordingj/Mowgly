

class Model(object):

    def score(self, val_generator, score_function):
        raise NotImplementedError

    def train(self, train_generator, loss_function):
        raise NotImplementedError