import pickle

# Load the pickle files for threshold 15
vsl_total = pickle.load(open("logger-vsl.pkl", "rb"))
novsl_total = pickle.load(open("logger-novsl.pkl", "rb"))
import re
import os
import decimal
import math

os.environ['INSTANCE_NAME'] = 'VSL_script_hell_die'
from global_settings import mds, vfs
import matplotlib.pyplot as plt
import numpy as np
import datetime

timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# Set font for the figure
# plt.rcParams.update({'font.size': 14})

_, ax = plt.subplots(figsize=(12, 6))
# Create graph for vsl
funcs = []
functions = []


def get_coefficients(interval):
    # Convert the string into a function array of terms
    terms = re.findall(r'([+-]?\s*\d+\.?\d*(?:e[+-]?\d+)?)(x\^\d+)?', interval['fitting_function'].replace(' ', ''))
    # For each element if x present, we extract exponent
    coefficients = [0] * (vfs['max_deg'] + 1)  # Initialize a list for coefficients
    for term in terms:
        coef = float(term[0])
        if term[1]:  # If there is an 'x' term
            exponent = int(term[1][2:])  # Get the exponent
            while len(coefficients) <= exponent:  # Expand the list if needed
                coefficients.append(0)
            # Assign the coefficient to the corresponding position in the list
            coefficients[exponent] = coef
        else:  # If there is no 'x' term, it's the constant term
            coefficients[0] = coef
    return coefficients


connection_points = []
labels = []
points = []
for element in vsl_total:
    interval = element['interval']
    print(interval)
    fitting_function_str = element['fitting_function']
    labels.append(fitting_function_str)
    [points.append(i) for i in element['fit_points']]

    coefficients = get_coefficients(element)
    fitting_function = np.poly1d(coefficients[::-1])

    funcs.append(fitting_function)

    if interval[1] != mds["domain_max_interval"]:
        connection_points.append(interval[1])
x_point = []
y_point = []
print(f"Fit points {points}")
for fit_point in points:
    print(f"FIT POINT: {fit_point}")
    x_point.append(fit_point[0])
    y_point.append(fit_point[1])

x = np.linspace(2600, 3900, 400, dtype=np.float128)

# Compute the y values for each function
y_values = [f(x) for f in funcs]
# Initialize the combined function as the first function
y = y_values[0]
y2 = y

# Define the transition function (sigmoid)
def transition(l, l_conn, width=20):
    return 1 / (1 + np.exp(-2 / width * (l - l_conn)))


# Iterate over the remaining functions
for i in range(1, len(funcs)):
    # Compute the transition values
    t = transition(x, connection_points[i - 1])

    # Update the combined function
    y = (1 - t) * y + t * y_values[i]
    print(t)

diff = y - y2
print(diff)
print(y.shape)
# Plot the functions and the combined function
labels = ['f1', 'f2', 'f1', 'f3', 'f1']
# labels = ['f1', 'f2', 'f1', 'f3', 'f1','ff','ff']
# colors = ['b', 'r', 'y', 'c','g','y','b','m']
colors = ['b', 'r', 'y', 'c', 'g']
print(connection_points)
for i in range(len(funcs)):
    if i == 0:
        plt.plot(x[x <= connection_points[i]], y_values[i][x <= connection_points[i]], colors[i] + '--',
                 label=labels[i])
    elif i == len(funcs) - 1:
        plt.plot(x[x >= connection_points[i - 1]], y_values[i][x >= connection_points[i - 1]], colors[i] + '--',
                 label=labels[i])
    else:
        plt.plot(x[(x >= connection_points[i - 1]) & (x < connection_points[i])],
                 y_values[i][(x >= connection_points[i - 1]) & (x < connection_points[i])], colors[i] + '--',
                 label=labels[i])

plt.plot(x, y, 'm--', label='combined')
for x_conn in connection_points:
    plt.axvline(x=x_conn, color='purple', linestyle=':')
plt.legend()
plt.show()