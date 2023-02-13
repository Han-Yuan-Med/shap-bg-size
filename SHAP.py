from __future__ import print_function
import shap
import torch
import numpy as np

# Calculate SHAP values and global variable rankings under each background dataset
# Background datasets are named as x repetition sample size y
# x stands for the simulation times and y means the sample number in each background dataset
# In our example, x=100 and y=50, which means 100 background datasets will be used.
# In each background dataset, there are 50 samples
model_optimal.eval()
orderlist = []
# i here stands for the simulation times: 100
# In this loop, 50 samples from train dataset will be randomly extracted as the background dataset
# Based on each background dataset, calculate the SHAP values and global ranking for following stability evaluation
for i in range(1, 101):
    # With a given seed, the sample will always draw the same rows.
    # If random_state is None or np.random, then a randomly-initialized RandomState object is returned.
    # Select 50 samples from train data randomly as the background data
    dataset_bg = dataset_train.sample(n=50, random_state=i)
    x_bg = dataset_bg.values[:, range(21)]
    x_bg_tensor = torch.FloatTensor(x_bg)
    e = shap.DeepExplainer(model_optimal, x_bg_tensor)
    shap_values = e.shap_values(x_valid_tensor)
    shap_values_1 = shap_values[1].squeeze()
    # Calculate the SHAP values of all samples in the valid dataset
    # Add the absolute values together to get the feature importance rankings
    shap_1_sum = np.absolute(shap_values_1).sum(axis=0)
    order_1 = np.argsort(-shap_1_sum) + 1
    orderlist.append(order_1)
    print("background dataset "+str(i)+" has been processed")
