
import numpy as np

from mowgly.utils import check_random_state

class WalkerAlias(object):
    """Implementation of Walker's alias method.
    This method generates a random sample from given probabilities
    :math:`p_1, \\dots, p_n` in :math:`O(1)` time.
    It is more efficient than :func:`~numpy.random.choice`.
    This class works on both CPU and GPU.
    Args:
        probs (float list): Probabilities of entries. They are normalized with
                            `sum(probs)`.
    See: `Wikipedia article <https://en.wikipedia.org/wiki/Alias_method>`_
    """

    def __init__(self, probs, random_state=None):

        probs = np.array(probs, np.float32)
        probs /= np.sum(probs)
        threshold = np.ndarray(len(probs), np.float32)
        values = np.ndarray(len(probs) * 2, np.int32)
        il, ir = 0, 0
        pairs = list(zip(probs, range(len(probs))))
        pairs.sort()
        for prob, i in pairs:
            p = prob * len(probs)
            while p > 1 and ir < il:
                values[ir * 2 + 1] = i
                p -= 1.0 - threshold[ir]
                ir += 1
            threshold[il] = p
            values[il * 2] = i
            il += 1
        # fill the rest
        for i in range(ir, len(probs)):
            values[i * 2 + 1] = 0

        assert((values < len(threshold)).all())
        self.threshold = threshold
        self.values = values
        self.random_state = check_random_state(random_state)


    def __call__(self, n):
        ps = self.random_state.uniform(0, 1, n)
        pb = ps * len(self.threshold)
        index = pb.astype(np.int32)
        left_right = (self.threshold[index] < (pb - index)).astype(np.int32)
        return self.values[index * 2 + left_right]