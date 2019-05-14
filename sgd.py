##############################################################################
## Code for Stochastic Gradient Descent
## Author Sunil Kakarlapudi
#
#Date: 3rd May 2019
###############################################################################

import numpy as np
import random
from matplotlib import pyplot as plt

#####################################################################################################################
# # Code to run sgd
#
# This is the main function that runs the Gradient Descent.
# It will take learningRate(eta), cost function, dimensions of the parameters(/weights) as input.
# It will find the local partial derivaties in each iteration, and move to a new set of parameters depending on the 
# gradiant and learning rate. It will stop moving further if it reaches a local minima.
# I added an upper limit of 100 on the iterations.
#
# We are taking the seed parameters between 
#####################################################################################################################
def sgd(eta, costfunction, dimensions):
    print("Starting SGD");
    print("\nLearning rate is ");
    print(eta)
    print("\nNumber of Parameters is ")
    print(dimensions)
    print("\nCost function is ")
    print(costfunction)
    print("\nTaking initial weights/parameters as a random number between [-1,1]")
    k = 0;
    parameterSequence = [];
    costSequence = [];
    # create seed parameters
    seed = [random.uniform(0,1) for _ in range(dimensions)];
    print("Initial Parameters:")
    print(seed);
    parameterSequence.append(seed);
    oldCost = 0;
    currentCost = 0;
    currentParameters = seed;
    while (((currentCost < oldCost) or k<2) and (k<100)):
        deltaParameters = iteration(eta, costfunction, currentParameters);
        oldCost = currentCost;
        currentCost = costfunction(deltaParameters)
        parameterSequence.append(deltaParameters);
        costSequence.append(currentCost);
        currentParameters = deltaParameters
        k = k + 1;
    print("\nFinal Parameters:");
    print(currentParameters);


    #######################################################################
    # Creating two plots.
    # 1) Cost at each iteration
    # 2) movement of parameters(first two dimensions) over the iterations.
    #######################################################################
    xarray = []; yarray = [];zarray=[];
    for dp in parameterSequence:
        if(len(dp)>1):
            xarray.append(dp[0]);
            yarray.append(dp[1]);

    xcost = [];ycost = [];itera = 0;
    for cst in costSequence:
        xcost.append(itera);
        ycost.append(cst);
        itera = itera + 1;

    plt.figure(0)
    plt.scatter(xarray,yarray);
    plt.title(" Change in Parameters Coordinates (First two Dimensions)");
    if(xarray[0]<xarray[1]):
        plt.xlabel("---------------- Direction of X Movement ------------>>");
    if(xarray[0]>xarray[1]):
        plt.xlabel("<<---------------- Direction of X Movement ------------");
        
    if(yarray[0]<yarray[1]):
        plt.ylabel("---------------- Direction of Y Movement ------------>>");
    if(yarray[0]>yarray[1]):
        plt.ylabel("<<---------------- Direction of Y Movement ------------");
    
    plt.plot(xarray,yarray);

    plt.figure(1);
    plt.title("Cost vs Iterations");
    plt.xlabel("Iterations");
    plt.ylabel("Cost");

    plt.scatter(xcost,ycost);
    plt.plot(xcost,ycost);
    plt.show();


##################################################################################
#
#  This function contains all the steps that we will have to perform in each iteration.
# It will take learningRate(eta), costFunction, and current parameters as input.
# And it will return the next step of the descent(new parameters) as output
#
##################################################################################
def iteration(eta, costfunction, parameters):
    temp = parameters;
    newParameters = [];
    i = 0;
    while i < len(parameters):
        temp = parameters;
        noMove = costfunction(temp)
        temp[i] = temp[i] + 0.01;
        upMove = costfunction(temp)
        temp[i] = temp[i] - 0.02;
        downMove = costfunction(temp)
        temp[i] = temp[i] + 0.01;
        if (noMove>upMove):
            newParameters.append(temp[i] + (eta * ((noMove-upMove)/0.01)))
        else:
            if (noMove>downMove):
                newParameters.append(temp[i] - (eta * ((noMove-downMove)/0.01)));
            else:
                newParameters.append(temp[i]);
        i = i + 1;
    return newParameters

##################################################################################
## Cost functions for testing.
# Created 2 cost functions for the purpose of testing.
# Ideally, the cost function itself will be defined by the program that is using 
# the gradient descent.
# As we are dealing with a 'Stochastic' Gradient Descent, The cost function should
# calcuate the cost as an average of the cost of all the data samples in one batch.
#
##################################################################################

#P is a linear vetcor of parameter values
def cost_squareSum(P):
    # assuming that the cost function is x1^2+x2^2+...
    cost = 0;
    for x in P:
        cost += x*x
    return cost

#P is a linear vetcor of parameter values
def cost_cosine(P):
    # assuming the cost function is cos(x1)+cos(x2)+...
    cost = 0;
    for x in P:
        cost += np.cos(x)
    return cost



################################################################################################################
# We are testing the SGD algorithm by running it on two cost functions with different parameters.
# calling the SGD algo with square sums cost function, and 2 dimensions.
################################################################################################################
print("\n\n-----------------------------------------------------------------------------------------------")
print("-----------------------------------------------------------------------------------------------\n\n")
print("Running the SGD for the cost function, Sum X squares with 5 dimensions.")
sgd(0.02, cost_squareSum, 5)
print("\n\n-----------------------------------------------------------------------------------------------")
print("-----------------------------------------------------------------------------------------------\n\n")
print("Running the SGD for the cost function, Sum of cosine(x) with 5 dimensions.")
sgd(0.75, cost_cosine, 5)