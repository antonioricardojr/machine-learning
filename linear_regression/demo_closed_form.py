import sys
from numpy import *

# getting the path data file.
if len(sys.argv) > 1:
    data_file = str(sys.argv[1])
else:
    data_file = '../data/income.csv'


def find_w1(points, x_mean, y_mean):
    '''Find the w1 coefficient'''
    num_obs = points.shape[0]

    numerator = 0.0
    denominator = 0.0
    for i in range(0, num_obs):

        diff_x = (points[i, 0] - x_mean)
        diff_y = (points[i, 1] - y_mean)

        numerator += diff_x * diff_y
        denominator += diff_x**2

    return numerator / denominator


def find_w0(w1, x_mean, y_mean):
    '''Find the w0 coefficient. It depends of w1'''
    return y_mean - (w1 * x_mean)


def coef_est(points):

    x_mean = mean(points[0])
    y_mean = mean(points[1])

    w1 = find_w1(points, x_mean, y_mean)
    w0 = find_w0(w1, x_mean, y_mean)

    return [w0, w1]


def compute_error_for_line_given_points(b, m, points):
    """Return the mean error"""
    return get_rss(b, m, points) / float(len(points))


def get_rss(b, m, points):
    """Return the Residual sum of squares"""
    total_error = 0
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        total_error += (y - (m * x + b)) ** 2

    return total_error


def run():
    points = genfromtxt(data_file, delimiter=',')

    num_columns = points.shape[1]
    num_obs = points.shape[0]

    print "Number os columns: " + str(num_columns)
    print "Number os observations: " + str(num_obs)

    [w0, w1] = coef_est(points)
    print "w0 = {0}, w1 = {1}, error = {2}".format(w0, w1, compute_error_for_line_given_points(w0, w1, points))

if __name__ == '__main__':
    run()
