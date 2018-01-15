import numpy as np

# get the hyothesis function with the theta parameter
def get_hypothesis(theta):
    def hypothesis(x):
        extended_x = np.concatenate((np.array([1]), x))
        theta_max = np.matrix(theta)
        extended_x_mat = np.matrix(extended_x)
        return (extended_x_mat * theta_max.T).item(0)
    return hypothesis

# compute the cost of hypothesis amount the x_data and y_data.The hypothesis function can be created by the function above
def compute_cost(hypothesis, x_data, y_data):
    sum = 0
    data_count = x_data.shape[0]
    for i in range(data_count):
        hypo_y = hypothesis(x_data[i])
        sum += (hypo_y - y_data[i].item(0)) ** 2
    return sum / data_count / 2

# compute the gradient of the hypothesis, for process the batch_gradient_decent
def compute_gradient(hypothesis, x_data, y_data):
    data_count = x_data.shape[0]
    extended_x_data = np.concatenate((np.ones(data_count).reshape((data_count, 1)), x_data), axis=1)
    result_count = extended_x_data.shape[1]
    result = np.zeros(result_count)
    for feature_index in range(0, result_count):
        sum = 0
        for item_index in range(0, data_count):
            factor = extended_x_data[item_index].item(feature_index)
            sum += (hypothesis(x_data[item_index]) - y_data[item_index]) * factor
        result[feature_index] = sum / data_count
    return result

# process the batch gradient decent algorithm, the step_callback function will be call in every step, if not None
def batch_gradient_decent(init_theta, learning_rate, x_data, y_data, max_step, step_callback = None):
    temp_theta = init_theta
    for i in range(0, max_step):
        hypothesis = get_hypothesis(temp_theta)
        gradient = compute_gradient(hypothesis, x_data, y_data)
        temp_theta -= learning_rate * gradient
        cost = compute_cost(hypothesis, x_data, y_data)
        if step_callback != None:
            step_callback(i, temp_theta, cost)
    return temp_theta

# process the normal equation algorithm
def normal_equation(x_data, y_data):
    X = np.matrix(np.concatenate((np.ones(x_data.shape[0]).reshape(x_data.shape[0], 1), x_data), axis=1))
    Y = np.matrix(y_data)
    theta = np.linalg.inv(X.T@X)@X.T@Y
    return theta.A1

# process the feature scaling
def feature_scaling(data):
    feature_num = data.shape[1]
    result = None
    for i in range(feature_num):
        col = data[:,i:i+1]
        max = col.max()
        min = col.min()
        mid = (max + min) / 2
        col = (col - mid) / (max - min) * 2
        if result is None:
            result = col
        else:
            result = np.concatenate((result, col), axis = 1)
    return result

