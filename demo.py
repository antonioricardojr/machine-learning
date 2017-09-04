
from numpy import *

def compute_error_for_given_point(b, m, points):
	

def step_gradient():
	#gradient descendent

def gradient_descendent_runner(points, starting_b, starting_m, learning_rate, num_iterations):
	b = starting_b
	m = starting_m

	for i in range(num_iterations):
		b, m = step_gradient(b, m, array(points), learning_rate)
	return [b, m]

def run():
	points = getfromtext('data.csv', delimiter = ',')

	#hyperparameters
	learning_rate = 0.0001
	#y = mx + b (slope formula)

	initial_b = 0
	initial_m = 0
	num_iterations = 1000
	[b, m] =  gradient_descendent_runner(points, initial_b, initial_m, learning_rate, num_iterations)
	print(b)
	print(m)






if __name__ == '__main__':
	run()