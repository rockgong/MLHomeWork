import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import linearregression as lr

# read the data from csv file
data = pd.read_csv('ex1data1.txt', header=None, names=['Population', 'Profit'])

# get the data for linear regression
col_num = data.shape[1]
x_data = data.iloc[:,0:col_num - 1]
y_data = data.iloc[:,col_num - 1 : col_num]

x_data = np.matrix(x_data)
y_data = np.matrix(y_data)

x_data = np.array(x_data)
y_data = np.array(y_data)

# data for plotting the step-cost figure for gradient decent
steps = []
costs = []
def step_callback(step, theta, cost):
   steps.append(step)
   costs.append(cost)

# process linear regression in two ways, gradient decent and normal equation
theta_gd = lr.batch_gradient_decent(np.zeros(x_data.shape[1] + 1), 0.02, x_data, y_data, 1000, step_callback)
theta_ne = lr.normal_equation(x_data, y_data)

# plot the data and the hypothesis function figure
hypo_gd = lr.get_hypothesis(theta_gd)
hypo_ne = lr.get_hypothesis(theta_ne)

def plot_hypothesis(hypo, min, max):
    x_plot_data = np.linspace(min, max, 100).reshape((100, 1))
    y_plot_data = np.array([hypo(x_plot_data[i]) for i in range(x_plot_data.shape[0])])
    plt.plot(x_plot_data, y_plot_data)

plt.figure()
plt.scatter(x_data, y_data)
plot_hypothesis(hypo_gd, x_data.min(), x_data.max())
plot_hypothesis(hypo_ne, x_data.min(), x_data.max())

# plot the step-cost figure for gradient decent
plt.figure()
plt.plot(np.array(steps), np.array(costs))
plt.show()
