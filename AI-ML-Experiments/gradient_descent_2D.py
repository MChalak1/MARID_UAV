# Gradient Descent/Ascent

# Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import sympy as sym
from IPython import display as dp
import matplotlib_inline.backend_inline
matplotlib_inline.backend_inline.set_matplotlib_formats('svg')

## Initializing The Functions

# Initializing Variable
x = sym.Symbol('x')
y = sym.Symbol('y')


# Initializing The 1st Function
fxy = x**2 + y**2 + 0.5*sym.cos(x*sym.pi) + 2*sym.cos(y*sym.pi/2)

print('fxy: ')
dp.display(fxy)

# Initialize The 2nd Function
gxy = 3*(1-x)**2*sym.exp(-x**2-(y+1)**2) -10*(x/5 - x**3 - y**5)*sym.exp(-x**2-y**2)-1/3*sym.exp(-(x+1)**2-y**2)

print('\n\ngxy: ')
dp.display(gxy)

# Computing Partial Derivatives
dfx_sym = sym.diff(fxy,x)
dfy_sym = sym.diff(fxy,y)
print('\n\ndfx: ')
dp.display(dfx_sym)
print('\n\ndfy: ')
dp.display(dfy_sym)

dgx_sym = sym.diff(gxy,x)
dgy_sym = sym.diff(gxy,y)
print('\n\ndgx: ')
dp.display(dgx_sym)
print('\n\ndgy: ')
dp.display(dgy_sym)


# Converting the functions into Numpy Functions using lambdify
fxy = sym.lambdify((x,y),fxy)
gxy = sym.lambdify((x,y),gxy)
dfx = sym.lambdify((x,y),dfx_sym)
dfy = sym.lambdify((x,y),dfy_sym)
dgx = sym.lambdify((x,y),dgx_sym)
dgy = sym.lambdify((x,y),dgy_sym)


## Parameters to be used later in the algorithm
# Introducing Algorithm tolerance
tol = 0.00001


## Plotting the functions using 4 subplots
# Domain Bounds
domain_bounds = [-5, 5]
x_range = np.linspace(-5,5,201)
y_range = np.linspace(-5,5,201)
plot_x, plot_y = np.meshgrid(x_range, y_range)

fig = plt.figure(figsize=(14, 8))

ax00 = fig.add_subplot(2, 2, 1, projection='3d')
ax01 = fig.add_subplot(2, 2, 2, projection='3d')
ax10 = fig.add_subplot(2, 2, 3)  # 2D Heat Map
ax11 = fig.add_subplot(2, 2, 4)  # 2D Heat Map

# Top row: 3D
ax00.set_title('Function 1')
ax00.set_xlabel('x'); ax00.set_ylabel('y'); ax00.set_zlabel('z')
ax00.plot_surface(plot_x, plot_y, fxy(plot_x, plot_y), cmap='viridis')

ax01.set_title('Function 2')
ax01.set_xlabel('x'); ax01.set_ylabel('y'); ax01.set_zlabel('z')
ax01.plot_surface(plot_x, plot_y, gxy(plot_x, plot_y), cmap='viridis')

# Bottom row: 2D imshow
ax10.set_title('Function 1')
ax10.set_xlabel('x'); ax10.set_ylabel('y')
ax10.imshow(fxy(plot_x, plot_y), extent=[x_range[0], x_range[-1], y_range[0], y_range[-1]],
            vmin=-5, vmax=5, origin='lower', cmap='viridis')

ax11.set_title('Function 2')
ax11.set_xlabel('x'); ax11.set_ylabel('y')
ax11.imshow(gxy(plot_x, plot_y), extent=[x_range[0], x_range[-1], y_range[0], y_range[-1]],
            vmin=-5, vmax=5, origin='lower', cmap='viridis')

plt.tight_layout()
plt.show()

## Gradient Descent for 2D Fxy

# Random Starting Point between -2 and 2
localmin = np.random.rand(2)*4-2    # Alternatively, we can specify Coordinates
startpnt = localmin[:]              # Making a copy of the localmin

# Learning Parameters
learning_rate = 0.01
training_epochs = 1000

# Gradient Descent
trajectory = np.zeros((training_epochs,2))

# Counting Trajectory slots used
slot = 0

for i in range(training_epochs):
  grad = np.array([ dfx(localmin[0],localmin[1]),
                  dfy(localmin[0],localmin[1])
                  ])
  localmin = localmin - learning_rate*grad
  trajectory[i,:] = localmin
  slot +=1

  # Adding a break if the function increases indefinitely
  if (localmin[0] < domain_bounds[0] or localmin[0] > domain_bounds[1] or
        localmin[1] < domain_bounds[0] or localmin[1] > domain_bounds[1]):
        print(f"Warning: Trajectory left domain at iteration {i}")
        break

  # Adding a break in case convergence occues at local extremum
  if np.linalg.norm(grad) < tol:
      print(f"Converged at iteration {i}, gradient norm={np.linalg.norm(grad):.2e}")
      break

print(localmin)
print(startpnt)
print(fxy(localmin[0],localmin[1]))

# Visualizing the result
plt.imshow(fxy(plot_x, plot_y), extent=[x_range[0], x_range[-1], y_range[0], y_range[-1]],
            vmin=-5, vmax=5, origin='lower', cmap='viridis')
plt.plot(startpnt[0], startpnt[1],'bs')
plt.plot(localmin[0], localmin[1],'ro')
plt.plot(trajectory[:slot,0], trajectory[:slot,1], 'r-')
plt.legend(['Starting Point', 'Local Minimum', 'Gradient Descent Trajectory'])
plt.colorbar()
plt.show()

## Gradient Descent 2D for Gxy

# Random Starting Point between -2 and 2
localmin = np.random.rand(2)*4-2    # Alternatively, we can specify Coordinates
startpnt = localmin[:]              # Making a copy of the localmin

# Learning Parameters
learning_rate = 0.01
training_epochs = 1000

# Gradient Descent
trajectory = np.zeros((training_epochs,2))

# Counting Trajectory slots used
slot = 0


for i in range(training_epochs):
  grad = np.array([ dgx(localmin[0],localmin[1]),
                  dgy(localmin[0],localmin[1])
                  ])
  localmin = localmin - learning_rate*grad
  trajectory[i,:] = localmin
  slot +=1

  # Adding a break if the function increases indefinitely
  if (localmin[0] < domain_bounds[0] or localmin[0] > domain_bounds[1] or
        localmin[1] < domain_bounds[0] or localmin[1] > domain_bounds[1]):
        print(f"Warning: Trajectory left domain at iteration {i}")
        break

  # Adding a break in case convergence occues at local extremum
  if np.linalg.norm(grad) < tol:
      print(f"Converged at iteration {i}, gradient norm={np.linalg.norm(grad):.2e}")
      break

  

print(localmin)
print(startpnt)
print(gxy(localmin[0],localmin[1]))

# Visualizing the result
plt.imshow(gxy(plot_x, plot_y), extent=[x_range[0], x_range[-1], y_range[0], y_range[-1]],
            vmin=-5, vmax=5, origin='lower', cmap='viridis')
plt.plot(startpnt[0], startpnt[1],'bs')
plt.plot(localmin[0], localmin[1],'ro')
plt.plot(trajectory[:slot,0], trajectory[:slot,1], 'r-')
plt.legend(['Starting Point', 'Local Minimum', 'Gradient Descent Trajectory'])
plt.colorbar()
plt.show()

## Gradient Ascent for Fxy

# Random Starting Point between -2 and 2
localmax = np.random.rand(2)*4-2    # Alternatively, we can specify Coordinates
startpnt = localmax[:]              # Making a copy of the localmax

# Learning Parameters
learning_rate = 0.01
training_epochs = 1000


# Gradient Ascent
trajectory = np.zeros((training_epochs,2))

# Counting Trajectory slots used
slot = 0


for i in range(training_epochs):
  grad = np.array([ dfx(localmax[0],localmax[1]),
                  dfy(localmax[0],localmax[1])
                  ])
  localmax = localmax + learning_rate*grad
  trajectory[i,:] = localmax
  slot +=1

  # Adding a break if the function increases indefinitely
  if (localmax[0] < domain_bounds[0] or localmax[0] > domain_bounds[1] or
        localmax[1] < domain_bounds[0] or localmax[1] > domain_bounds[1]):
        print(f"Warning: Trajectory left domain at iteration {i}")
        break

  # Adding a break in case convergence occues at local extremum
  if np.linalg.norm(grad) < tol:
      print(f"Converged at iteration {i}, gradient norm={np.linalg.norm(grad):.2e}")
      break
  

print(localmax)
print(startpnt)
print(fxy(localmax[0],localmax[1]))

# Visualizing the result
plt.imshow(fxy(plot_x, plot_y), extent=[x_range[0], x_range[-1], y_range[0], y_range[-1]],
            vmin=-5, vmax=5, origin='lower', cmap='viridis')
plt.plot(startpnt[0], startpnt[1],'bs')
plt.plot(localmax[0], localmax[1],'ro')
plt.plot(trajectory[:slot,0], trajectory[:slot,1], 'r-')
plt.legend(['Starting Point', 'Local Max', 'Gradient Ascent Trajectory'])
plt.colorbar()
plt.show()

## Gradient Ascent for Gxy

# Random Starting Point between -2 and 2
localmax = np.random.rand(2)*4-2    # Alternatively, we can specify Coordinates
startpnt = localmax[:]              # Making a copy of the localmax

# Learning Parameters
learning_rate = 0.01
training_epochs = 1000


# Gradient Ascent
trajectory = np.zeros((training_epochs,2))

# Counting Trajectory slots used
slot = 0


for i in range(training_epochs):
  grad = np.array([ dgx(localmax[0],localmax[1]),
                  dgy(localmax[0],localmax[1])
                  ])
  localmax = localmax + learning_rate*grad
  trajectory[i,:] = localmax
  slot +=1

  # Adding a break if the function increases indefinitely
  if (localmax[0] < domain_bounds[0] or localmax[0] > domain_bounds[1] or
        localmax[1] < domain_bounds[0] or localmax[1] > domain_bounds[1]):
        print(f"Warning: Trajectory left domain at iteration {i}")
        break

  # Adding a break in case convergence occues at local extremum
  if np.linalg.norm(grad) < tol:
      print(f"Converged at iteration {i}, gradient norm={np.linalg.norm(grad):.2e}")
      break

  

print(localmax)
print(startpnt)
print(gxy(localmax[0],localmax[1]))

# Visualizing the result
plt.imshow(gxy(plot_x, plot_y), extent=[x_range[0], x_range[-1], y_range[0], y_range[-1]],
            vmin=-5, vmax=5, origin='lower', cmap='viridis')
plt.plot(startpnt[0], startpnt[1],'bs')
plt.plot(localmax[0], localmax[1],'ro')
plt.plot(trajectory[:slot,0], trajectory[:slot,1], 'r-') # Modified to start plotting from index 1
plt.legend(['Starting Point', 'Local Max', 'Gradient Ascent Trajectory'])
plt.colorbar()
plt.show()
