# Parametric LR for Binary Classification

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from IPython import display as dp
import sympy as sym
import matplotlib_inline.backend_inline
matplotlib_inline.backend_inline.set_matplotlib_formats('svg')

"""
Binary classification with PyTorch ANN.

Creates two 2D Gaussian clusters (class A at (1,1), class B at (5,1)) and trains
a feedforward network (2→h→1 with ReLU) to classify points by color.
Uses BCEWithLogitsLoss (no sigmoid needed) and a 0 threshold to assign predicted labels.

Components:
- Data: 100 points per class, blurred by a configurable factor
- Model: Linear(2, h) → ReLU → Linear(h, 1) where h is configurable (default: 1)
- Loss: BCEWithLogitsLoss (combines sigmoid + BCE, outputs logits)
- Output: logits (raw scores); predictions = logits > 0
- Accuracy: 100 * mean((predictions > 0) == labels)

Experiments:
- Parametric learning rate sweep (0.001 to 0.1)
- Meta-experiment: average accuracy over 50 random initializations
- Comparison across hidden units (h=1, 2, 5)
- Observations: Higher learning rates and more hidden units improve accuracy
"""

## Generating Data Function

def generate_data(nPerClust = 100, blur = 1, show_graph = True):

  """
  Creates two 2D Gaussian clusters (class A at (1,1), class B at (5,1)) to be fed
  as input to a small feedforward net to classify points by color.

  Parameters:
  - nPerClust (int): Number of points per class. Default is 100.
  - blur (float): How spread out the data will be. Default is 1.
  - show (bool): Whether to show the plot. Default is True.

  Returns: data, labels
  """

  # Creating The Data:
  n = nPerClust
  br = blur        # How spread out the data will be
  A = [1, 1]      # Centering Data A around coord- (1,1)
  B = [5, 1]      # Centering Data B around coord- (5,1)

  # Generating Data:
  # 'a' and 'b' are lists of 2 arrays. One for abscissa and another for ordinate
  # The coordinates are randomly scattered by the factor "blur"
  a = [A[0]+np.random.randn(n)*br, A[1]+np.random.randn(n)*br]
  b = [B[0]+np.random.randn(n)*br, B[1]+np.random.randn(n)*br]

  # True Labels
  # Vertically stacking data labels fo A and B
  labels_np = np.vstack((np.zeros((n, 1)), np.ones((n, 1))))

  # Concatanate into a Matrix:
  # A 200x2 matrix is generated. Col 1 abscissa and Col 2 ordinate
  # 0:99 Data "a" // 100:199 Data "b"
  data_np= np.hstack((a, b)).T

  # Convert to Tensors:
  data = torch.tensor(data_np).float()
  labels = torch.tensor(labels_np).float()

  # Showing the true data:
  fig = plt.figure(figsize=(5, 5))

  # Plotting the first 100 points as blue squares in the left half of the plane:
  plt.plot(data[np.where(labels==0)[0],0], data[np.where(labels==0)[0],1],'bs')

  # Plotting the second 100 points as black circles in the right half of the plane:
  plt.plot(data[np.where(labels==1)[0],0], data[np.where(labels==1)[0],1],'ko')
  if show_graph:
    plt.title('Data')
    plt.xlabel('dim 1')
    plt.ylabel('dim 2')
    plt.show()

  return data, labels

## Model Function

def model(data, learningRate, hidden_units = 1):
  """
  Creates and initializes a binary classification neural network with loss function and optimizer.
  
  Builds a feedforward network: Linear(2, h) → ReLU → Linear(h, 1), where h is the number
  of hidden units. Uses BCEWithLogitsLoss (no sigmoid needed) and SGD optimizer.
  
  Parameters:
  -----------
  data : torch.Tensor
      Input data tensor (shape: N×2). Used only for type conversion, not for training.
  learningRate : float
      Learning rate for SGD optimizer (e.g., 0.01, 0.1).
  hidden_units : int, optional
      Number of hidden units in the network. Default is 1.
      More units = more model capacity (h=1: 3 params, h=5: 15 params).
  
  Returns:
  --------
  ANNclassify : nn.Sequential
      The neural network model (2→h→1 architecture with ReLU).
  lossFunction : nn.BCEWithLogitsLoss
      Binary cross-entropy loss with logits (combines sigmoid + BCE).
  optimizer : torch.optim.SGD
      Stochastic gradient descent optimizer configured with the given learning rate.
  """
  
  # Building the Model
  data = data.float()
  h = hidden_units

  ANNclassify = nn.Sequential(
      nn.Linear(2, h),     # Input layer (2 input, h output)
      nn.ReLU(),           # Activation Function
      nn.Linear(h, 1),     # Output Unit
  )

  ANNclassify

  # Learning Rate
  learningRate = learningRate

  # Loss Function
  lossFunction = nn.BCEWithLogitsLoss()       # Binary Cross Entropy loss

  # Optimizer (Implementing Gradient Descent)
  optimizer = torch.optim.SGD(ANNclassify.parameters(), lr = learningRate)


  return ANNclassify, lossFunction, optimizer

## Model Trainer Function

def train_model(ANNclassify, lossFunction, optimizer, data, labels, \
                numepochs = 1000, show_graph=False, print_accuracy = False):
  """
  Trains a binary classification neural network using gradient descent.
  
  Performs forward pass, computes loss, backpropagates gradients, and updates
  model parameters for the specified number of epochs. Returns training history
  and final accuracy.
  
  Parameters:
  -----------
  ANNclassify : nn.Module
      The neural network model to train.
  lossFunction : nn.Module
      Loss function (typically BCEWithLogitsLoss).
  optimizer : torch.optim.Optimizer
      Optimizer configured with model parameters and learning rate.
  data : torch.Tensor
      Input features (shape: N×2).
  labels : torch.Tensor
      Binary labels (shape: N×1, values 0 or 1).
  numepochs : int, optional
      Number of training epochs. Default is 1000.
  show_graph : bool, optional
      If True, plots loss curve and highlights final loss. Default is False.
  print_accuracy : bool, optional
      If True, prints final accuracy to console. Default is False.
  
  Returns:
  --------
  losses : torch.Tensor
      Loss value at each epoch (shape: numepochs).
  predictions : torch.Tensor
      Final model predictions (logits) on training data.
  totalacc : float
      Final accuracy percentage (0-100).
  
  Note:
  -----
  Accuracy is computed as: 100 * mean((predictions > 0) == labels)
  This uses logits threshold of 0 (equivalent to probability threshold of 0.5).
  """

  # Training The Model:
  numepochs = numepochs
  ann = ANNclassify
  data = data.float()
  labels = labels.float()
  losses = torch.zeros(numepochs)

  for epochi in range(numepochs):

    # Forward pass
    y_hat = ann(data)

    # Computing the loss
    loss = lossFunction(y_hat, labels)
    losses[epochi] = loss

    # Backprop
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


  # Computing Predictions:
  predictions = ANNclassify(data)     # Result of BCEWL function y_hat

 
  # Total Accuracy:
  totalacc = 100*torch.mean(((predictions > 0)==labels).float())

  if show_graph:
      # Plotting the Losses:
      plt.plot(losses.detach(), 'o', markerfacecolor='w', linewidth=.1)
      plt.plot(epochi, losses[-1].detach(), 'o', markerfacecolor='r', markersize=7)
      plt.xlabel('Epochs')
      plt.ylabel('Loss')
      plt.title(f'Losses over Epochs, Final Accuracy {totalacc}%')
      plt.show()

  # Report Accuracy:
  if print_accuracy:
    print(f'Accuracy: {totalacc}%')
  return losses, predictions, totalacc

## Code Test With 1 Iteration:

# Generating The Data:
data, labels = generate_data(show_graph=True)

# Initializing The Model:
ANNclassify, lossFunction, optimizer = model(data, learningRate=0.01, hidden_units = 1)

# Training The Model:
losses, predictions, totalacc = train_model(ANNclassify, lossFunction,\
                                            optimizer, data, labels,\
                                            numepochs=1000, show_graph=True)

## Experimenting The Model for Variable Learning Rates

# Setting Learning Rates
learningrates = np.linspace(0.001, 0.1, 40)
numepochs = 1000

# Initializing Results
accByLr = []        # Accuracy per Learning Rate
allLosses = np.zeros((len(learningrates), numepochs))

# Running The Experiment
for i,lr in enumerate(learningrates):
  
  # Initializing The Model:
  ANNclassify, lossFunction, optimizer = model(data, lr, hidden_units = 1)

  # Train The Model:
  losses, predictions, totalacc = train_model(ANNclassify,lossFunction, \
                                              optimizer, data, labels, numepochs,\
                                              show_graph=False)

  # Storing All Results
  allLosses[i,:] = losses.detach()
  accByLr.append(totalacc)

# Plot the Labeled Data:
fig, ax = plt.subplots(1,2, figsize=(12,4))

ax[0].plot(learningrates, accByLr, 's-')
ax[0].set_xlabel('Learning Rate')
ax[0].set_ylabel('Accuracy')
ax[0].set_title('Accuracy by Learning Rate')

ax[1].plot(allLosses.T)
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Loss')
ax[1].set_title('Losses by Learning Rate')
plt.show()

# Proportion of runs where accuracy exceeded 70%
sum(torch.tensor(accByLr) > 70)/len(accByLr)

## Meta-Experiment
Running a Meta-Experiment on the previous one conducted 50 times

# Setting up outer loop.
numepochs = 500
meta_j = 50
learningrates = np.linspace(0.001, 0.1, 40)   # Setting Learning Rates

accMeta = np.zeros((meta_j, len(learningrates)))  


for j in range(meta_j):

  # Running The Experiment
  for i,lr in enumerate(learningrates):
    
    # Initializing The Model:
    ANNclassify, lossFunction, optimizer = model(data, lr, hidden_units = 1)

    # Train The Model:
    losses, predictions, totalacc = train_model(ANNclassify, lossFunction, \
                                                optimizer, data, labels,\
                                                numepochs, show_graph=False)

    # Storing All Results
    accMeta[j,i] = totalacc
  

# Plotting The Results Averaged over experiments
plt.plot(learningrates, np.mean(accMeta, axis=0), 's-') # Avrage of acc along each column
plt.xlabel('Learning Rate')
plt.ylabel('Accuracy')
plt.title('Accuracy by Learning Rate')
plt.show()

## Mega-Meta Experiment
Expanding on Meta-experiments by looping through different hiddent units values.
-  **NOTE** The code takes ~20 mins

fig, ax = plt.subplots(figsize=(10, 6))

for h in [1, 2, 5]:
  # Setting up outer loop.
  numepochs = 500
  meta_j = 50
  learningrates = np.linspace(0.001, 0.1, 40)   # Setting Learning Rates

  accMeta = np.zeros((meta_j, len(learningrates)))  


  for j in range(meta_j):

    # Running The Experiment
    for i,lr in enumerate(learningrates):
      
      # Initializing The Model:
      ANNclassify, lossFunction, optimizer = model(data, lr, hidden_units = h)

      # Train The Model:
      losses, predictions, totalacc = train_model(ANNclassify, lossFunction, \
                                                  optimizer, data, labels,\
                                                  numepochs, show_graph=False)

      # Storing All Results
      accMeta[j,i] = totalacc
    

  # Plotting Results:
  ax.plot(learningrates, np.mean(accMeta, axis=0), 's-', label=f'Hidden Units: {h}')

ax.set_xlabel('Learning Rate')
ax.set_ylabel('Accuracy')
ax.set_title('Accuracy by Learning Rate (Different Hidden Units)')
ax.legend()
ax.grid(True, alpha=0.3)
plt.show()

## Observations
"""
 Across repeated runs with different random initializations, accuracy tended to increase with higher learning rates in the tested range (0.001–0.1). 
 For this simple, low‑dimensional, linearly separable problem and fixed number of epochs, larger learning rates help the optimizer move quickly toward a good decision boundary, 
 so they converge to high accuracy more reliably than very small learning rates, which make only tiny parameter updates and often underfit within the allotted epochs.
 This trend is specific to this toy setup (data, model size, epoch budget, LR range); in more complex or noisier problems, too-high learning rates would typically cause instability or divergence rather than improved performance.

 Additionally, increasing hidden units (h=1 → h=5) consistently improved accuracy across all learning rates, indicating that the additional model capacity helps capture the nonlinear decision boundary needed to separate the overlapping Gaussian clusters. 
 The improvement was most pronounced at moderate to high learning rates (0.01–0.1), where the larger models converged more reliably.
 """
