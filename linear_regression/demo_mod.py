import sys
from numpy import *

##getting the path data file.
if len(sys.argv) > 1:
	data_file = str(sys.argv[1])
else:
	data_file = 'income.csv'


rss_file_output = open('./data/rss_output.csv', 'w')
rss_file_output.write('iteration;rss\n') ##file header

def write_rss(i,rss):
	rss_file_output.write(str(i) + ';' + str(rss) + '\n')


def step_gradient(b_current, m_current, points, learning_rate):
	b_gradient = 0
	m_gradient = 0
	N = float(len(points))

	for i in range(0, len(points)):
		x = points[i, 0]
		y = points[i, 1]

		b_gradient += -(2/N) * (y - ((m_current * x) + b_current))
		m_gradient += -(2/N) * x * (y - ((m_current * x) + b_current))

	new_b = b_current - (learning_rate * b_gradient)
	new_m = m_current - (learning_rate * m_gradient)

	return [new_b, new_m]

def compute_error_for_line_given_points(b, m, points):
	"""Return the mean error"""
	return get_rss(b, m, points) / float(len(points))

def get_rss(b, m, points):
	"""Return the Residual sum of squares"""
	total_error = 0
	for i in range(0,len(points)):
		x = points[i, 0]
		y = points[i, 1]
		total_error += (y - (m*x + b)) **2
		
	return total_error 

def gradient_descent_runner(points, starting_b, starting_m, learning_rate, num_iterations):
	b = starting_b
	m = starting_m

	for i in range(num_iterations):
		b, m = step_gradient(b, m, array(points), learning_rate)

		rss = get_rss(b, m, points)
		write_rss(i,rss)
		print "RSS: {0}".format(rss)
	return [b, m]

def run():
	points = genfromtxt(data_file, delimiter = ',')


	learning_rate = input("Learning Rate: ")
	#hyperparameters
	if not float(learning_rate):
		learning_rate = 0.0001
	#y = mx + b (slope formula)

	initial_b = 0 #initial y-intercept guess
	initial_m = 0 #initial slope guess

	num_iterations = input("Number of Iterations: ")
	if not int(num_iterations):
		num_iterations = 1000

	print "Starting gradient descent at b = {0}, m = {1}, error = {2}".format(initial_b, initial_m, compute_error_for_line_given_points(initial_b, initial_m, points))
	print "Running..."
	[b, m] = gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iterations)
	print "After {0} iterations b = {1}, m = {2}, error = {3}".format(num_iterations, b, m, compute_error_for_line_given_points(b, m, points))
	rss_file_output.close()

if __name__ == '__main__':
    run()