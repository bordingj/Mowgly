
class Step(object):

    def fit(self, **inputs):
        raise NotImplementedError

    def forward(self, **inputs):
        raise NotImplementedError

    def fit_forward(self, **inputs):
        self.fit(**inputs)
        return self.forward(**inputs)