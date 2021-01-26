# -*- coding: utf-8 -*-
"""
Linear regression by gradient descent.
"""


from numpy import *
import os


def compute_error(b,m,points):
	totalError = 0
	for i in range(0,len(points)):
		x = points[i,0]
		y = points[i,1]
		totalError += (y - (m*x+b))**2
	return totalError/float(len(points))



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


# determine when the search finds a solution
def gradient_descent_runner(points, initial_m, initial_b, learning_rate, max_num_iter, tol = 1e-3):
	b = initial_b
	m = initial_m
	num_iter = 0
	new_error = 0
	old_error = compute_error(initial_b, initial_m, points)
	tot_error = 999
	while (tot_error > tol and num_iter < max_num_iter):
		num_iter = num_iter + 1
		new_b, new_m = step_gradient(b, m, array(points), learning_rate)
		old_error = compute_error(b, m, points)
		new_error = compute_error(new_b, new_m, points)
		tot_error = abs(new_error - old_error)
		b = new_b
		m = new_m
	return [new_b, new_m, num_iter]


def run():
	os.chdir('/some_dir/')
	points = np.genfromtxt('data.csv', delimiter=',')

	learning_rate = 0.0001
	# y = mx + b
	initial_b = 0
	initial_m = 0
	num_iterations = 1000
	max_num_iter = 1000

	print ("Starting gradient descent at b = {0}, m = {1}, error = {2}".format(initial_b, initial_m, compute_error(initial_b, initial_m, points)))
	print ("Running...")

	[final_b, final_m, num_iter] = gradient_descent_runner(points, initial_m, initial_b, learning_rate, num_iterations)
	print ("After {0} iterations b = {1}, m = {2}, error = {3}".format(num_iter, final_b, final_m, compute_error(final_b, final_m, points)))



if __name__ == '__main__':
	run()
