import seaborn as sns
import matplotlib.pyplot as plt


class DataHandler:

    # converting dataframe into lists
    @staticmethod
    def listify(dataframe):
        return dataframe.values.tolist()

    # converting dataframe into array
    @staticmethod
    def getArray(dataframe):
        return dataframe.to_array()

    # counting the number of separable classes in a given data frame
    @staticmethod
    def get_dataframe_feature_count(dataframe, feature):
        return dataframe[feature].value_counts()

    # setting graph background to a darker, high-contrast overtone
    @staticmethod
    def set_to_dark_mode():
        sns.set_style("darkgrid")
        sns.color_palette("pastel")
        print("set to 'dark-mode'")

    # setting graph background to a brighter, distinct color palette
    @staticmethod
    def set_to_light_mode():
        sns.set_style("whitegrid")
        sns.color_palette("bright")
        print("set to 'light-mode'")

    # plotting histogram of given distinct features in a class set
    @staticmethod
    def plot_histogram_overlap(dataframe, feature_string):
        sns.FacetGrid(dataframe, hue='class', height=5, palette='colorblind').map(sns.histplot, feature_string)\
            .add_legend(title="Class", adjust_subtitles=True)
        plt.show()
        print("histogram printed to screen")

    # plotting a 2 x 2 histogram of given distinct features in a class set
    @staticmethod
    def plot_histogram_2x2(dataframe, title='Measurement Frequency'):
        dataframe.plot.hist(subplots=True, layout=(2, 2), figsize=(7, 7), bins=16, alpha=0.5, title=title)
        plt.show()
        print("2x2 histogram printed to screen")

    # plotting histogram and scatter plot of given distinct feature combinations in a class set
    @staticmethod
    def plot_combinations(dataframe):
        sns.pairplot(dataframe, hue='class', diag_kind="hist", height=2.5, palette="colorblind",
                     plot_kws={'alpha': 0.6}, diag_kws={'alpha': 0.55, 'bins': 16}, markers=["o", "s", "D"])
        plt.show()
        print("Pair plot array printed to screen")

    # visualizing scatter plot of a given set of x and y features in a data frame. Classifier must be specified
    @staticmethod
    def scatterplot(dataframe, x_feature, y_feature, classifier='class'):
        sns.scatterplot(x=x_feature, y=y_feature, hue=classifier, data=dataframe)
        print("Scatter plot printed to screen")

    # visualizing line plot of a given set of x and y features in a data frame
    @staticmethod
    def lineplot(x_values, y_values, title='', x_label='', y_label=''):
        sns.lineplot(x=x_values, y=y_values).set_title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.show()

    # pretty printed divider in terminal window
    @staticmethod
    def divider(string=''):
        print("========================" + string + "========================")
