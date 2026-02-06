# Manipulating Parameters

# Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
from IPython import display as dp
import sympy as sym
import matplotlib_inline.backend_inline
matplotlib_inline.backend_inline.set_matplotlib_formats('svg')

# Initializing the Function:
x = sym.Symbol('x')

fx = sym.sin(x)*sym.exp(-x**2*0.05)   # Function f(x)
dfx = sym.diff(fx,x)                  # Derivative of f(x)


dp.display(fx, dfx)                   # Displaying functions


# Converting Functions to numpy
f = sym.lambdify(x, fx, 'numpy')
df = sym.lambdify(x, dfx, 'numpy')

# Plotting the function and its derivative:
x = np.linspace(-2*np.pi, 2*np.pi, 401)

plt.plot(x, f(x), x, df(x))
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid()
plt.legend(['f(x)', 'f\'(x)'])

## Gradient Descent 1D

# Random Starting Point
localmin = np.random.choice(x,1)

# Learning Parameters
learning_rate = 0.01
training_epochs = 1000

# Run through the training
for i in range(training_epochs):
  grad = df(localmin)
  localmin = localmin - learning_rate*grad

# Plotting Results:
plt.plot(x, f(x), x, df(x),'--')
plt.plot(localmin, f(localmin), 'ro')
plt.plot(localmin, df(localmin), 'ro')
plt.legend(['f(x)', 'f\'(x)', 'Local Minimum'])
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title(f'Local Minimum x= {localmin[0]}')
plt.grid()

## Running Parametric Experiments

## Experiment 1: Varying Starting Location

startloc = np.linspace(-5,5,50)
finalres = np.zeros(len(startloc))          # Storing local minima


# Looping over Starting Points

# Getting the index and the startloc(idx)
for idx, localmin in enumerate(startloc):   # localmin = startloc(idx)

  # Running the training
  for i in range(training_epochs):
    grad = df(localmin)
    localmin = localmin - learning_rate*grad

  # Storing guesses
  finalres[idx] = localmin

# Plotting the results:

fig = plt.figure(figsize=(10,6))

ax0 = fig.add_subplot(121)
ax1 = fig.add_subplot(122)

ax0.set_title('local Minima / Initial Guess')
ax0.set_xlabel('Starting Location')
ax0.set_ylabel('Local Minimum (x_min)')


ax1.set_title('f(localminima) / Initial Guess')
ax1.set_xlabel('Starting Location')
ax1.set_ylabel('f(x_min)')

ax0.plot(startloc, finalres, 'bo-')
ax0.grid()
ax1.plot(startloc, f(finalres), 'ro-')
ax1.grid()


## Experiment 2: Systematically Varying The Learning Rate

learning_rate = np.linspace(1e-10,0.1,50)
finalres = np.zeros(len(learning_rate))          # Storing local minima

# Looping over learning rate and implementing GD:
for idx, lr in enumerate(learning_rate):

  # Fixing the initial guess to better understand behavior with respect to lr
  localmin = 0

  # Run through the training:
  for i in range(training_epochs):
    grad = df(localmin)
    localmin = localmin - lr*grad
  
  # Storing the final guess:
  finalres[idx] = localmin

# Plotting the results:
plt.plot(learning_rate, finalres, 'bo-')
plt.xlabel('Learning Rate')
plt.ylabel('Local Minimum (x_min)')
plt.grid()

## Experiment 3: Correlation Between Learning Rate & Training Epochs

# Setting-up parameters:
learning_rate = np.linspace(1e-10,0.1,50)
training_epochs = np.round(np.linspace(10,500,40))  

# Initializing matrix to store the results
finalres = np.zeros((len(learning_rate), len(training_epochs)))

# Running the training:
# Looping over lr
for idxl, lr in enumerate(learning_rate):

  # Looping over the training epochs
  for idxe, tepoch in enumerate(training_epochs):

    # Fixing the initial guess to better understand
    localmin = 0

    # Run through the training

    for i in range(int(tepoch)):
      grad = df(localmin)
      localmin = localmin - lr*grad

    # Storing the final guess
    finalres[idxl, idxe] = localmin


# Plotting the results:
fig, ax = plt.subplots(figsize=(10,5))
plt.imshow(finalres, extent=[learning_rate[0], learning_rate[-1], \
                             training_epochs[0], training_epochs[-1]],\
           aspect='auto', origin='lower', vmin=-1.45, vmax = -1.2)
plt.colorbar()
plt.ylabel('Training Epochs')
plt.xlabel('Learning Rate')
plt.title('Final Guess')
plt.show()

# Another visual:
plt.plot(learning_rate,finalres)
plt.xlabel('Learning Rate')
plt.ylabel('Final function estimate')
plt.title('Each Line -> Training Epoch N')
plt.grid()
plt.show()


