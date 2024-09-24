import pandas as pd
import torch

data = pd.read_csv('data\house_tiny.csv')
inputs = data.iloc[: ,0:2]
outputs = data.iloc[: ,2]
inputs = inputs.fillna(inputs.mean())   
inputs = pd.get_dummies(inputs, dummy_na=True)
print(inputs)
X = torch.tensor(inputs.to_numpy(dtype=float))
Y = torch.tensor(outputs.to_numpy(dtype=float))
print(X,Y)