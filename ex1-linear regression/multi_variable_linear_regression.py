import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import linearregression as lr

# read the data from csv file
data = pd.read_csv('ex1data2.txt', header=None, names=['Size', 'Bedrooms', 'Price'])

# get the data for linear regression
col_num = data.shape[1]
x_data = data.iloc[:,0:col_num - 1]
y_data = data.iloc[:,col_num - 1 : col_num]

x_data = np.matrix(x_data)
y_data = np.matrix(y_data)

x_data = np.array(x_data)
y_data = np.array(y_data)

# process the feature scaling
x_data_scaled = lr.feature_scaling(x_data)
y_data_scaled = lr.feature_scaling(y_data)

# the data for plotting step-cost figure for gradient decent
steps = []
costs = []
def step_callback(step, theta, cost):
   steps.append(step)
   costs.append(cost)

# process the linear regression algorithm.
theta_gd = lr.batch_gradient_decent(np.zeros(x_data.shape[1] + 1), 0.01, x_data_scaled, y_data_scaled, 1000, step_callback)
theta_ne = lr.normal_equation(x_data_scaled, y_data_scaled)

# view result
print(theta_gd)
print(theta_ne)

hypothesis_gd = lr.get_hypothesis(theta_gd)
hypothesis_ne = lr.get_hypothesis(theta_ne)

cost_gd = lr.compute_cost(hypothesis_gd, x_data_scaled, y_data_scaled)
cost_ne = lr.compute_cost(hypothesis_ne, x_data_scaled, y_data_scaled)

print(cost_gd)
print(cost_ne)

# plot the step-cost figure for gradient decent
plt.figure()
plt.plot(np.array(steps), np.array(costs))
plt.show()
