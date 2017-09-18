

class ConsoleLogger(object):

    def log_value(self, name, value, iter):
        """

        :param name:
        :type name: str
        :param value:
        :type value: float
        :param iter:
        :type iter: int
        :return: None
        """
        msg = "{0} : {1}, {2:2.10f}".format(int(iter), name, float(value) )
        print(msg)


class SilentLogger(object):

    def log_value(self, name, value, iter):
        """

        :param name:
        :type name: str
        :param value:
        :type value: float
        :param iter:
        :type iter: int
        :return: None
        """
        pass