import argparse
from pathlib import Path
import flwr as fl
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import time

# ================================
# Paths
# ================================
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)
CLIENT_WEIGHTS_DIR = MODEL_DIR / "client_saved"
CLIENT_WEIGHTS_DIR.mkdir(exist_ok=True)

# ================================
# Simple PyTorch audio CNN for spectrograms
# ================================
class SimpleAudioCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 32 * 11, 256),  # adjust if input size changes
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.net(x)

# ================================
# helpers: PyTorch state_dict <-> numpy list
# ================================
def get_parameters(model):
    return [val.cpu().numpy() for _, val in model.state_dict().items()]

def set_parameters(model, parameters):
    state_dict = {k: torch.tensor(v) for k, v in zip(model.state_dict().keys(), parameters)}
    model.load_state_dict(state_dict, strict=True)

# ================================
# Load audio dataset
# ================================
def load_local_audio_dataset(client_dir, target_sr=22050, n_mels=128, width=44):
    X, Y = [], []
    base = client_dir
    classes = sorted([p.name for p in base.iterdir() if p.is_dir()])
    cls2idx = {c: i for i, c in enumerate(classes)}

    for cls in classes:
        for f in (base / cls).glob('*'):
            waveform, sr = torchaudio.load(f)
            # Mono conversion
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            # Resample if needed
            if sr != target_sr:
                waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)(waveform)

            # ================================
            # Safe MelSpectrogram call
            # ================================
            try:
                mel_transform = torchaudio.transforms.MelSpectrogram(sample_rate=target_sr, n_mels=n_mels)
                mel = mel_transform(waveform)
            except TypeError:
                # fallback for older torchaudio
                from torchaudio.functional import melspectrogram
                mel = melspectrogram(waveform, sample_rate=target_sr, n_mels=n_mels)

            db_transform = torchaudio.transforms.AmplitudeToDB()
            db = db_transform(mel)

            # Take mean across channels
            spec = db.mean(dim=0)

            # Pad or trim to fixed width
            if spec.shape[1] < width:
                pad = torch.zeros(n_mels, width - spec.shape[1])
                spec = torch.cat([spec, pad], dim=1)
            else:
                spec = spec[:, :width]

            spec = spec.unsqueeze(0)  # (1, n_mels, width)
            X.append(spec.numpy())
            Y.append(cls2idx[cls])

    X = np.stack(X) if len(X) > 0 else np.zeros((0, 1, n_mels, width))
    Y = np.array(Y)
    return X, Y, classes

# ================================
# Flower PyTorch client
# ================================
class PyTorchClient(fl.client.NumPyClient):
    def __init__(self, model, X, Y, device, client_id, local_epochs=1, batch_size=8):
        self.model = model
        self.X = X
        self.Y = Y
        self.device = device
        self.client_id = client_id
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)

    def get_parameters(self, config=None):
        return get_parameters(self.model)

    def fit(self, parameters, config):
        print(f"[Client {self.client_id}] Starting local training...")
        set_parameters(self.model, parameters)
        self.model.train()

        idx = np.arange(len(self.X))
        for epoch in range(self.local_epochs):
            np.random.shuffle(idx)
            for i in range(0, len(idx), self.batch_size):
                batch_idx = idx[i:i + self.batch_size]
                xb = torch.tensor(self.X[batch_idx], dtype=torch.float32).to(self.device)
                yb = torch.tensor(self.Y[batch_idx], dtype=torch.long).to(self.device)
                self.optimizer.zero_grad()
                out = self.model(xb)
                loss = self.criterion(out, yb)
                loss.backward()
                self.optimizer.step()
            print(f"[Client {self.client_id}] Epoch {epoch+1}/{self.local_epochs} done, last batch loss: {loss.item():.4f}")

        # Save model with timestamp
        ts = int(time.time())
        model_path = CLIENT_WEIGHTS_DIR / f"audio_model_{self.client_id}_{ts}.pth"
        torch.save(self.model.state_dict(), model_path)
        print(f"[Client {self.client_id}] Model saved at {model_path}")

        return get_parameters(self.model), len(self.X), {"loss": float(loss.item())}

    def evaluate(self, parameters, config):
        set_parameters(self.model, parameters)
        self.model.eval()
        correct, total, total_loss = 0, 0, 0.0
        with torch.no_grad():
            xb = torch.tensor(self.X, dtype=torch.float32).to(self.device)
            yb = torch.tensor(self.Y, dtype=torch.long).to(self.device)
            out = self.model(xb)
            loss = self.criterion(out, yb)
            total_loss += loss.item()
            pred = out.argmax(dim=1)
            correct += (pred == yb).sum().item()
            total += yb.size(0)
        acc = correct / total if total > 0 else 0.0
        print(f"[Client {self.client_id}] Eval done: loss={total_loss:.4f}, acc={acc:.4f}")
        return float(total_loss), total, {"accuracy": float(acc)}

# ================================
# Main
# ================================
def main(server: str, client_id: str, local_epochs: int = 1):
    client_audio_dir = Path("CLIENT_DATA") / "audio"
    if not client_audio_dir.exists():
        raise RuntimeError(f"No audio data for client at {client_audio_dir}")

    X, Y, classes = load_local_audio_dataset(client_audio_dir)
    if len(classes) == 0:
        raise RuntimeError("No audio classes found")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleAudioCNN(len(classes)).to(device)
    client = PyTorchClient(model, X, Y, device, client_id, local_epochs=local_epochs)

    # Federated learning
    fl.client.start_numpy_client(
        server_address=server,
        client=client
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--server', type=str, required=True)
    parser.add_argument('--client_id', type=str, required=True)
    parser.add_argument('--local_epochs', type=int, default=1)
    args = parser.parse_args()
    main(args.server, args.client_id, args.local_epochs)

