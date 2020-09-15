import pandas as pd
from KNN import KNearestNeighbors
from DataHandler import DataHandler


def main():
    dataframe_train = pd.read_csv("data\\iris_train.csv")
    dataframe_val = pd.read_csv("data\\iris_dev.csv")
    dataframe_test = pd.read_csv("data\\iris_test.csv")

    dl = DataHandler()

    ''' Question a.i '''
    dl.divider(" Question a.i ")
    print("Training Set Class Count ::")
    print(dl.get_dataframe_feature_count(dataframe_train, "class"))

    ''' Question a.ii '''
    dl.divider(" Question a.ii ")
    dl.plot_histogram_2x2(dataframe_train)
    dl.plot_histogram_overlap(dataframe_train, 'sepal_length')
    dl.plot_histogram_overlap(dataframe_train, 'sepal_width')
    dl.plot_histogram_overlap(dataframe_train, 'petal_length')
    dl.plot_histogram_overlap(dataframe_train, 'petal_width')

    ''' Question a.iii '''
    dl.divider(" Question a.iii ")
    dl.plot_combinations(dataframe_train)
    print()

    ''' Question b.i '''
    dl.divider(" Question b.i ")
    print("Further information is provided in the KNN.py file")

    ''' Question b.ii '''
    dl.divider(" Question b.ii ")
    print("Further information is provided in the main.py file in the main method")

    dataframe_train_list = dl.listify(dataframe_train)
    dataframe_val_list = dl.listify(dataframe_val)
    dataframe_test_list = dl.listify(dataframe_test)

    k_accuracy = []

    '''
     similarity settings range from 0 - 3
     0 - l0 norm
     1 - l1 or one norm
     2 - l2 or euclidean distance
     l3 - cosine_similarity
    '''
    similarity_setting = 2

    for k in range(1, 20, 2):
        knn = KNearestNeighbors(k=k)

        prediction_set = []

        for data_row in dataframe_val_list:
            neighbors = knn.getNearestNeighbors(dataframe_train_list, data_row, similarity_setting)
            result = knn.getResponse(neighbors)
            prediction_set.append(result)

        acc = knn.evaluate_accuracy(dataframe_test_list, prediction_set)
        print("K :: " + str(k) + ", Accuracy :: " + repr("{:.2f}".format(acc)) + "%")
        k_accuracy.append((k, acc))

    # if (similarity_setting == 0):
    #     print("l0")
    # elif (similarity_setting == 1):
    #     print("l1")
    # elif (similarity_setting == 2):
    #     print("l2")
    # else:
    #     print("cos")

    dl.lineplot([x[0] for x in k_accuracy], [x[1] for x in k_accuracy], "K classification accuracy", "K Value", "Accuracy")


    ''' Question b.iii '''
    dl.divider(" Question b.iii ")
    print("Further information is provided in the main.py file in the main method and discussed in the Word file.")


if __name__ == "__main__":
    main()
