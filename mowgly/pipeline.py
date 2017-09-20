
from mowgly.step import Step


class Pipeline(object):
    """
    This is similar to sklearn.Pipeline much more minimalistic and flexible.
    """
        
    def __init__(self, steps):
        """
        :param steps: list of instances of children of mowgly.step.Step
        """
        self._validate_steps(steps)
        self.steps = steps
        self.n_steps = len(steps)
        
    def _validate_steps(self, steps):
        assert isinstance(steps, list)
        assert len(steps) > 0
        for step in steps:
            assert isinstance(step, Step)

    def fit(self, **inputs):
        """
        
        Args:
            **inputs: keyword arguments feed through the pipeline
        
        return:
            self
        """

        for step in self.steps[:-1]:
            inputs = step.fit_forward(**inputs)
        self.steps[-1].fit(**inputs)
        
        return self
    
    def forward(self, **inputs):
        """
        
        Args:
            **inputs: keyword arguments to be passed through pipeline
        
        return:
            final_outputs.
        """

        for step in self.steps:
            inputs = step.forward(**inputs)

        return inputs