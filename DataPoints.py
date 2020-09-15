import numpy as np


class DataPoint(object):

    def __init__(self, feature):
        """
        Return a Datapoint object whose attributes are given by "features"
        """
        self.sepal_length = feature['sepal_length']
        self.sepal_width = feature['sepal_width']
        self.petal_length = feature['petal_length']
        self.petal_width = feature['petal_width']
        self.name = feature['name']

    def feature_vector(self):
        """
        Return feature vector as a numpy array
        """
        return np.array([self.sepal_length, self.sepal_width, self.petal_length, self.petal_width, self.name])

    def __str__(self):
        """
        print(object) uses this method
        Define it in the way you want to see the object (e.g. Height:27.0, Weight:50.0, Label:1)
        """
        return "Name:{}, Sepal Height:{}, Sepal Length:{}, Petal Height:{}, Petal Length:{}".format(
            self.name, self.sepal_length, self.sepal_width, self.petal_length, self.petal_width)
