import torch
import torch.nn.functional as F
import pandas as pd
import torch.nn as nn
import numpy as np
import random
import sklearn.metrics as metrics

# Define the artificial neural networks for modeling; Here we use a three-layer MLP as an example
class MLP(torch.nn.Module):

    def __init__(self):
        super(MLP, self).__init__()  #
        self.fc1 = torch.nn.Linear(21, 128)
        self.fc2 = torch.nn.Linear(128, 64)
        self.fc3 = torch.nn.Linear(64, 2)

    def forward(self, m):
        m = F.relu(self.fc1(m))
        m = F.relu(self.fc2(m))
        m = F.softmax(self.fc3(m), dim=1)
        return m


# Define the model training function
def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} \tLoss: {:.6f}'.format(
                epoch, loss.item()))


# Open train data for model development
with open("train_data.csv") as f:
    dataset_train = pd.read_csv(f)
f.close()
# Open valid data for model interpretation based on SHAP
# It should be noted that the valid data here is for model interpretation.
# Such data for interpretation can also be called valid data which is different from the common notion
# The interpretation data can come from training data or validation data
with open("valid_data.csv") as f:
    dataset_valid = pd.read_csv(f)
f.close()

train_dataset = pd.DataFrame(dataset_train).values
valid_dataset = pd.DataFrame(dataset_valid).values
# Extract features in training data and valid data
x_train = train_dataset[:, range(21)]
x_valid = valid_dataset[:, range(21)]
# Extract labels in training data and valid data
y_train = train_dataset[:, 21]
y_valid = valid_dataset[:, 21]
# Transform features to tensors
x_train_tensor = torch.FloatTensor(x_train)
x_valid_tensor = torch.FloatTensor(x_valid)
y_train_tensor = torch.LongTensor(y_train)
train_dataset = torch.utils.data.TensorDataset(x_train_tensor, y_train_tensor)
# Prepare train data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=2048, shuffle=True)
# Set weights using inverse probability to prevent loss function from focusing on majority categories
weights = [0.089, 0.911]
class_weights = torch.FloatTensor(weights)
# Set criterion
criterion = nn.CrossEntropyLoss(weight=class_weights)
# Initialize MLP
seed = 1234
model = MLP()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
auroc_value = np.zeros(100)

for i in range(1, 501):
    train(i)
    i_tmp = i % 5
    if i_tmp == 0:
        model.eval()
        # calculate the fpr and tpr for all thresholds of the classification
        y_pred_tmp = model(x_valid_tensor)
        preds_tmp = y_pred_tmp[:, 1]
        preds1_tmp = preds_tmp.detach().numpy()
        fpr, tpr, threshold = metrics.roc_curve(y_valid, preds1_tmp)
        roc_auc = metrics.auc(fpr, tpr)
        auroc_value[(i // 5)-1] = roc_auc
        # store the model
        PATH = "model after " + str(i // 5) + "epochs training.pt"
        torch.save(model, PATH)


# Output roc on valid dataset and select the optimal model.
# The selected optimal model will be interpreted by the valid data and SHAP
del model
model_optimal = torch.load("model after " + str(np.argmax(auroc_value)+1) + "epochs training.pt")
