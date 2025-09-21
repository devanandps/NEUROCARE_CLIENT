import argparse
from pathlib import Path
import flwr as fl
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
from sklearn.preprocessing import LabelEncoder

# ================================
# Paths
# ================================
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)
CLIENT_MODELS_DIR = MODEL_DIR / "client_saved"
CLIENT_MODELS_DIR.mkdir(exist_ok=True)

# ================================
# Simple Feedforward Model
# ================================
class TabularNN(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.net(x)

# ================================
# Helpers: PyTorch <-> Flower
# ================================
def get_parameters(model):
    return [val.cpu().numpy() for _, val in model.state_dict().items()]

def set_parameters(model, parameters):
    state_dict = {k: torch.tensor(v) for k, v in zip(model.state_dict().keys(), parameters)}
    model.load_state_dict(state_dict, strict=True)

# ================================
# Load & Sanitize CSV Data
# ================================
def load_csv_dataset(csv_path: Path):
    df = pd.read_csv(csv_path)

    # Features: F001-F080 (all columns starting with F)
    feature_cols = [c for c in df.columns if c.startswith("F")]
    if not feature_cols:
        raise ValueError("No feature columns found (expected columns starting with 'F').")

    X = df[feature_cols].values.astype(np.float32)

    # Labels: AI_Label (0-4) OR probabilities P_Normal..P_Tumour
    if "AI_Label" in df.columns:
        Y = df["AI_Label"].values
    else:
        prob_cols = [c for c in df.columns if c.startswith("P_")]
        if not prob_cols:
            raise ValueError("CSV must contain 'AI_Label' or probability columns starting with 'P_'.")
        Y = df[prob_cols].values.argmax(axis=1)

    # Encode labels if they are strings
    if Y.dtype == object or isinstance(Y[0], str):
        le = LabelEncoder()
        Y = le.fit_transform(Y)

    # Ensure integer type for classification
    Y = Y.astype(np.int64)

    # Replace NaNs in features
    X = np.nan_to_num(X, nan=0.0)

    # Normalize features to [0,1]
    max_val = np.max(X)
    if max_val > 0:
        X = X / max_val

    num_classes = len(np.unique(Y))

    # Debug info
    print("✅ Data loaded and sanitized")
    print(f"Features shape: {X.shape}, Labels shape: {Y.shape}, Num classes: {num_classes}")
    print("Sample features (first row):", X[0][:10])
    print("Sample labels:", Y[:5])
    print("Any NaNs in features?", np.isnan(X).any())
    print("Any NaNs in labels?", np.isnan(Y).any())

    return X, Y, num_classes

# ================================
# Flower PyTorch Client
# ================================
class TabularClient(fl.client.NumPyClient):
    def __init__(self, model, X, Y, device, client_id, local_epochs=1, batch_size=16):
        self.model = model
        self.X = X
        self.Y = Y
        self.device = device
        self.client_id = client_id
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

    def get_parameters(self, config=None):
        return get_parameters(self.model)

    def fit(self, parameters, config):
        print(f"[Client {self.client_id}] Starting local training...")
        set_parameters(self.model, parameters)
        self.model.train()

        idx = np.arange(len(self.X))
        last_loss = 0.0
        for epoch in range(self.local_epochs):
            np.random.shuffle(idx)
            for i in range(0, len(idx), self.batch_size):
                batch_idx = idx[i:i+self.batch_size]
                xb = torch.tensor(self.X[batch_idx], dtype=torch.float32).to(self.device)
                yb = torch.tensor(self.Y[batch_idx], dtype=torch.long).to(self.device)
                self.optimizer.zero_grad()
                out = self.model(xb)
                loss = self.criterion(out, yb)
                loss.backward()
                self.optimizer.step()
                last_loss = loss.item()
            print(f"[Client {self.client_id}] Epoch {epoch+1}/{self.local_epochs} done, last loss={last_loss:.4f}")

        ts = int(time.time())
        model_path = CLIENT_MODELS_DIR / f"es_model_{self.client_id}_{ts}.pth"
        torch.save(self.model.state_dict(), model_path)
        print(f"[Client {self.client_id}] Model saved: {model_path}")

        return get_parameters(self.model), len(self.X), {"loss": float(last_loss)}

    def evaluate(self, parameters, config):
        set_parameters(self.model, parameters)
        self.model.eval()
        with torch.no_grad():
            xb = torch.tensor(self.X, dtype=torch.float32).to(self.device)
            yb = torch.tensor(self.Y, dtype=torch.long).to(self.device)
            out = self.model(xb)
            loss = self.criterion(out, yb)
            pred = out.argmax(dim=1)
            acc = (pred == yb).float().mean().item()
        print(f"[Client {self.client_id}] Eval done: loss={loss.item():.4f}, acc={acc:.4f}")
        return float(loss.item()), len(self.X), {"accuracy": float(acc)}

# ================================
# Main
# ================================
def main(server: str, client_id: str, csv_file: str, local_epochs: int = 1):
    csv_path = Path("CLIENT_DATA") / "es" / csv_file
    if not csv_path.exists():
        raise RuntimeError(f"CSV file not found: {csv_path}")

    X, Y, num_classes = load_csv_dataset(csv_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TabularNN(input_dim=X.shape[1], num_classes=num_classes).to(device)

    client = TabularClient(model, X, Y, device, client_id, local_epochs=local_epochs)

    fl.client.start_numpy_client(
        server_address=server,
        client=client,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", type=str, required=True)
    parser.add_argument("--client_id", type=str, required=True)
    parser.add_argument("--csv_file", type=str, required=True, help="CSV file name inside CLIENT_DATA/es/")
    parser.add_argument("--local_epochs", type=int, default=1)
    args = parser.parse_args()
    main(args.server, args.client_id, args.csv_file, args.local_epochs)

