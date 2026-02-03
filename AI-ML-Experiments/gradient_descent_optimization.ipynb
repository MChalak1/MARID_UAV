import numpy as np
import matplotlib.pyplot as plt
import math
import sympy as sym
from IPython import display as dp
import matplotlib_inline.backend_inline
matplotlib_inline.backend_inline.set_matplotlib_formats('svg')


## Gradient Descent in 1D
### The Given Function
# Defining the function (as a function)
pi_val = math.pi # Use for numerical calculations
x_sym = sym.symbols('x') # Define x as a symbolic variable for SymPy


# Define the symbolic expression for f(x) for differentiation with SymPy
# Use sym.cos and sym.pi with the symbolic variable x_sym
fx_symbolic = sym.cos(2*sym.pi*x_sym) + x_sym**2
dp.display(fx_symbolic)

# Calculate the symbolic derivative
deriv_symbolic = sym.diff(fx_symbolic, x_sym)
dp.display(deriv_symbolic)

# Calculate the second derivative 
deriv2_symbolic = sym.diff(deriv_symbolic, x_sym)
dp.display(deriv2_symbolic)

# Create a numerical function from the symbolic functions
fx = sym.lambdify(x_sym, fx_symbolic, 'numpy')
deriv = sym.lambdify(x_sym, deriv_symbolic, 'numpy')
deriv2 = sym.lambdify(x_sym, deriv2_symbolic, 'numpy')

# Plotting the function and its derivatives

# Defining the domain:
x = np.linspace(-2, 2, 2001)

# Plotting
plt.plot(x, fx(x), x, deriv(x))
plt.xlim(x[[0,-1]])
plt.grid()
plt.legend(['f(x)', 'f\'(x)'])
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend(['y','dy'])
plt.show()

### Learning Algorithm
# Starting Point
localmin = np.random.choice(x,1)    # From vector x, randomly choose 1 value
print('First Estimate: ', localmin)


### Running the Algo & Storing the DATA
# Starting Point
localmin = np.random.choice(x,1)    # From vector x, randomly choose 1 value
print('First Estimate: ', localmin)

# learning parameters
learning_rate = .01        # Small learning steps
training_epochs = 150     # Number of iterations

# Run through training and store all results:
modelparams = np.zeros((training_epochs,2))
for j in range(training_epochs):
  grad = deriv(localmin)
  localmin = localmin - learning_rate*grad
  modelparams[j,0] = localmin[0]
  modelparams[j,1] = grad[0]

# Plot the gradient over iterations

fig, ax = plt.subplots(1,2, figsize=(12,4))

for i in range(2):
  ax[i].plot(modelparams[:,i], 'o-')
  ax[i].set_xlabel('Iteration')
  ax[i].set_title(f'Final Estimated Minimum: {localmin[0]:.5f}')

ax[0].set_ylabel('Local Minimum')
ax[1].set_ylabel('Derivative')

plt.show()

# The smaller the training rate the higher the number of epochs needed to reach
# the local minima.

# We can clearly see that number of iterations exceded what's needed. Next,
# we'll adopt a different approach

### Using an Intended Gradient Value
# We'll try and run a similar code, but for a
# defined value of the derivative intended to reach

# Starting Point
localmin = np.random.choice(x,1)      # From vector x, randomly choose 1 value
initial_guess = localmin[0]
print('First Estimate: ', localmin)
print('F(localmin): ', fx(localmin))
print('Derivative: ', deriv(localmin))
print('Second Derivative: ', deriv2(localmin))

# learning parameters
grad_lim = 0.0001                     # Intended grad
learning_rate = .01                   # Small learning steps

grad = deriv(localmin)

# Escaping potential local max
while abs(grad[0]) <= 1e-12 and deriv2(localmin)[0]<0:
  localmin = localmin + math.e**-3
  grad = deriv(localmin)


# Run through training and store all results:
# Initialize modelparams as a list to dynamically store results
modelparams_list = []
max_iters = 20000
j = 0
while abs(grad[0]) >= grad_lim and j < max_iters:
  modelparams_list.append([localmin[0],grad[0]]) # Append to the list
  grad = deriv(localmin)
  localmin = localmin - learning_rate*grad
  j+=1

# Convert the list to a NumPy array after the loop for subsequent plotting
modelparams = np.array(modelparams_list)

localmin

# Plot the gradient over iterations

fig, ax = plt.subplots(1,2, figsize=(12,4))

for i in range(2):
  ax[i].plot(modelparams[:,i], 'o-')
  ax[i].set_xlabel('Iteration')
  
ax[0].set_title(f'Parameter (x) Over Iterations -> x_min: {localmin[0]:.5f}')
ax[1].set_title('Gradient Over Iterations')
ax[0].set_ylabel('Local Minimum')
ax[1].set_ylabel('Derivative')

plt.show()

print('Number of iterations: ', j)
plt.plot(x, fx(x), x, deriv(x))
plt.plot(initial_guess, fx(initial_guess), 'yo')
plt.plot(localmin, fx(localmin), 'ro')
plt.plot(localmin, deriv(localmin), 'ro')
plt.plot(modelparams[:, 0], fx(modelparams[:, 0]), 'k-', alpha=1.0)
plt.xlim(x[[0,-1]])
plt.grid()
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend([
  'f(x)', 'df',
  'initial guess', 'f(x) at min', 'df at min',
  'optimization path'
])
plt.title('Empirical Local Minimum: %s'%localmin[0])
plt.show()
