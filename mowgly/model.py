

class Model(object):

    def score(self, val_generator):
        raise NotImplementedError

    def loss(self, train_generator):
        raise NotImplementedError

    def set_to_inference_mode(self):
        raise NotImplementedError

    def set_to_training_mode(self):
        raise NotImplementedError