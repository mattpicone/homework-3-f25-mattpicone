import numpy as np
import matplotlib.pyplot as plt


def norm_histogram(histogram):
    """
    takes a list of counts and converts to a list of probabilities, outputs the probability list.
    :param histogram: a numpy ndarray object
    :return: list
    """

    num_samples = np.sum(histogram)
    probabilities = histogram / num_samples

    return probabilities

def compute_j(histogram, bin_width, num_samples):
    """
    takes list of counts, uses norm_histogram function to output the histogram of probabilities,
    then calculates compute_j for one specific bin width (reference: histogram.pdf page19)
    :param histogram: list
    :param bin_width: float
    :param num_samples: int
    :return: float
    """

    probabilities = norm_histogram(histogram)
    all_probabilities = np.sum(probabilities ** 2)

    #formula from histrogram.pdf page 19
    j = (2 / ((num_samples - 1) * bin_width)) - ((num_samples + 1) / ((num_samples - 1) * bin_width)) * all_probabilities
    return j

def sweep_n(data, min_val, max_val, min_bins, max_bins):
    """
    find the optimal bin
    calculate compute_j for a full sweep [min_bins to max_bins]
    please make sure max_bins is included in your sweep

    The variable "data" is the raw data that still needs to be "processed"
    with matplotlib.pyplot.hist to output the histogram

    You must utilize the variables (data, min_val, max_val, min_bins, max_bins)
    in your code for 'sweep_n' to determine the correct input to the function 'matplotlib.pyplot.hist',
    specifically the values to (x, bins, range).
    Other input variables of 'matplotlib.pyplot.hist' can be set as default value.

    :param data: list
    :param min_val: int
    :param max_val: int
    :param min_bins: int
    :param max_bins: int
    :return: list
    """

    errors = []
    num_samples = len(data)

    for bins in range(min_bins, max_bins + 1):
        histogram, fluff, fluff2 = plt.hist(data, bins, range = (min_val, max_val))
        bin_width = (max_val - min_val) / bins
        j = compute_j(histogram, bin_width, num_samples)
        errors.append(j)


    return errors


def find_min(l):
    """
    takes a list of numbers and returns the three smallest number in that list and their index.
    return a dict i.e.
    {index_of_the_smallest_value: the_smallest_value, index_of_the_second_smallest_value: the_second_smallest_value, ...}

    For example:
        A list(l) is [14,27,15,49,23,41,147]
        Then you should return {0: 14, 2: 15, 4: 23}

    :param l: list
    :return: dict: {int: float}
    """

    smallest_nums = {}

    count = 0

    while count < 3:
        smallest = np.min(l)
        index = l.index(smallest)
        smallest_nums[index] = smallest
        l[index] = 10000000000000000000000 #very large number so it is not the smallest
        count += 1

    return smallest_nums


# ============================== P2 ==================================


import scipy.stats as stats
import numpy as np


def get_data(filename):
    return np.loadtxt(filename)


def get_coordinates(data, each_dist):
    # Part B
    """
    calculates the QQ plot given an array of data and a name of a distribution
    outputs a tuple of 2 numpy arrays from the output of the QQ plot
    :param data: np.ndarray
    :param each_dist: str
    :return: (np.ndarray, np.ndarray)
    """
    # Your code starts here...

    from_plot = stats.probplot(data, dist = each_dist)
    coordinates = from_plot[0]
    x = coordinates[0]
    y = coordinates[1]

    return x, y


def calculate_distance(x, y):
    # Part B
    """
    calculates the projected distance between x and y
    returns the distance as a float
    :param x: float
    :param y: float
    :return: float
    """
    middle = (x + y) / 2

    dist_x = x - middle
    dist_y = y - middle

    distance = ((dist_x ** 2) + (dist_y ** 2)) ** 0.5


    return distance


def find_dist(data):
    # Part B
    """
    from a dictionary of distribution names and their respective errors, finds the distribution having the minimum value
    outputs the minimum value and the name of the distribution
    :param data: dict: {str: float}
    :return: (str, float)
    """

    #big number so it is definitely not the smallest
    smallest_error = 100000000000
    found_index = 0

    for x in data:
        val = data[x]
        if val < smallest_error:
            smallest_error = val
            found_index = x

    return found_index, smallest_error


def main(data_file):
    """
    Input a csv file and return distribution type, the error corresponding to the distribution type (e.g. return ('norm', 0.32))
    :param: *.csv file name (str)
    :return: (str, float)
    """
    data = get_data(data_file)
    dists = ("norm", "expon", "uniform", "wald")
    sum_err = [0] * 4
    for ind, each_dist in enumerate(dists):
        X, Y = get_coordinates(data, each_dist)
        for x, y in zip(X, Y):
            sum_err[ind] += calculate_distance(x, y)
    return find_dist(dict(zip(dists, sum_err)))


if __name__ == "__main__":
    data = np.loadtxt("input.txt")  # reads data from input.txt
    lo = min(data)
    hi = max(data)
    bin_l = 1
    bin_h = 100
    js = sweep_n(data, lo, hi, bin_l, bin_h)
    """
    the values bin_l and bin_h represent the lower and higher bound of the range of bins.
    They will change when we test your code and you should be mindful of that.
    """
    print(find_min(js))
  ############### Uncomment for P2 #################

    for each_dataset in [
        "sample_norm.csv",
        "sample_expon.csv",
        "sample_uniform.csv",
        "sample_wald.csv",
        "distA.csv",
        "distB.csv",
        "distC.csv",
    ]:
        print(main(each_dataset))
