
import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class MLP(nn.Module):

    def __init__(self, no_features, no_hidden, no_labels):
        super().__init__()
        self.mlp_stack = nn.Sequential(
            nn.Linear(no_features, no_hidden),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(no_hidden, no_hidden),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(no_hidden, no_hidden),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(no_hidden, no_labels),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.mlp_stack(x)

def preprocess(df, test_size=0.2, random_state=42):
    df_train, y_train, df_test, y_test = split_dataset(df, ["filename"], test_size, random_state)
    X_train_scaled, X_test_scaled = preprocess_dataset(df_train, df_test)
    return X_train_scaled, y_train, X_test_scaled, y_test

class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

