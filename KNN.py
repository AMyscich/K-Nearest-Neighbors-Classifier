import numpy as np
import math


class KNearestNeighbors():

    # setting the initial value of k neighbors
    def __init__(self, k=5):
        self.k = k

    # the l0 is calculated using the reference: https://medium.com/@montjoile/l0-norm-l1-norm-l2-norm-l-infinity-norm-7a7d18a4f40c#:~:text=For%20example%2C%20the%20L0%20norm,then%20the%20login%20is%20successful.
    # this calculates the sum of non zero vectors in a data set.
    @staticmethod
    def zero_norm(vec1, vec2):
        sum = 0
        v1, v2 = np.array(vec1), np.array(vec2)
        for i in range(len(vec1) - 1):
            if((float(v1[i]) - float(v2[i])) != 0):
                sum += 1
        return sum

    # the l1 or one norm distance is calculated using the reference: https://medium.com/@montjoile/l0-norm-l1-norm-l2-norm-l-infinity-norm-7a7d18a4f40c#:~:text=For%20example%2C%20the%20L0%20norm,then%20the%20login%20is%20successful.
    # this calculates the sum of the magnitudes of the vectors in a space
    @staticmethod
    def one_norm(vec1, vec2):
        dist = 0
        v1, v2 = np.array(vec1), np.array(vec2)
        for i in range(len(vec1) - 1):
            dist += abs(float(v1[i]) - float(v2[i]))
        return dist

    # the euclidean distance is calculated using the reference: https://medium.com/@montjoile/l0-norm-l1-norm-l2-norm-l-infinity-norm-7a7d18a4f40c#:~:text=For%20example%2C%20the%20L0%20norm,then%20the%20login%20is%20successful.
    # this ultimately finds the shortest distance from one point to another.
    @staticmethod
    def euclidean_distance(vec1, vec2):
        dist = 0
        v1, v2 = np.array(vec1), np.array(vec2)
        for i in range(len(vec1) - 1):
            dist += (float(v1[i]) - float(v2[i])) ** 2
        return np.sqrt(dist)

    # cosine similarity is calculated using the reference: https://www.machinelearningplus.com/nlp/cosine-similarity/
    # Once calculated, the value is subtracted from 1 to get the most "votes" in get Reference.
    @staticmethod
    def cosine_similarity(vec1, vec2):
        v1, v2 = np.array(vec1), np.array(vec2)
        dot = sum((float(a) * float(b)) for a, b in zip(v1[:-1], v2[:-1]))
        norm_a = math.sqrt(sum((float(a) * float(a)) for a in v1[:-1]))
        norm_b = math.sqrt(sum((float(b) * float(b)) for b in v2[:-1]))

        # Cosine similarity
        cos_sim = float(dot) / (norm_a * norm_b)
        # reversing cosine percentage (to a lower vote priority listing)
        return (1 - cos_sim)

    # accuracy of data point classifier is assessed between a given data set and prediction set.
    @staticmethod
    def evaluate_accuracy(dataset, prediction_set):
        correct_count = 0
        for i in range(len(dataset)):
            if (dataset[i][-1] == prediction_set[i]):
                correct_count += 1
        return (float(correct_count) / len(dataset)) * 100.0

    # getting the nearest neighbor of a given point
    # several selections of calculated distances are determined by the flag marker given in the main file
    # returning neighbors in sorted order for later "tallying" in getResponse
    def getNearestNeighbors(self, train_set, test_row, flag=2):
        distances = []
        random_index = np.random.permutation(np.array(train_set).shape[0])
        train_set_new = np.array([train_set[random_index[i]] for i in range(1, 90)]).tolist()
        for train_row in train_set_new:
            if(flag == 0):
                dist = self.zero_norm(test_row, train_row)
            elif(flag == 1):
                dist = self.one_norm(test_row, train_row)
            elif(flag == 2):
                dist = self.euclidean_distance(test_row, train_row)
            else:
                dist = self.cosine_similarity(test_row, train_row)

            distances.append((train_row, dist))
        distances.sort(key=lambda x: x[1])
        neighbors = []
        for i in range(self.k):
            neighbors.append(distances[i][0])
        return neighbors

    # getting classification from nearest neighbor votes
    def getResponse(self, neighbors):
        classVotes = {}
        for neighbor in neighbors:
            response = neighbor[-1]
            if (response in classVotes):
                classVotes[response] += 1
            else:
                classVotes[response] = 1
        sortedVotes = sorted(classVotes.items(), key=lambda x: x[1], reverse=True)
        return sortedVotes[0][0]
