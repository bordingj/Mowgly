
from mowgly.step import Step


class Pipeline(object):
    """
    This is similar to sklearn.Pipeline much more minimalistic and flexible.
    """
    
    def _validate_steps(self, steps):
        assert isinstance(steps, list)
        assert len(steps) > 0
        for step in steps:
            assert isinstance(step, Step)
        
    def __init__(self, steps):
        self._validate_steps(steps)
        self.steps = steps
        self.n_steps = len(steps)
        
        
    def fit(self, **inputs):
        """
        
        Args:
            **inputs: keyword arguments feed through the pipeline
        
        return:
            self
        """
        
        for i in range(self.n_steps - 1):
            inputs = self.steps[i].fit_forward(**inputs)
        self.steps[i + 1].fit(**inputs)
        
        return self
        
    
    def forward(self, **inputs):
        """
        
        Args:
            **inputs: keyword arguments to be passed through pipeline
        
        return:
            final_outputs.
        """
        
        for i in range(self.n_steps - 1):
            inputs = self.steps[i].forward(**inputs)
        final_outputs = self.steps[i + 1].predict(**inputs)
        
        return final_outputs