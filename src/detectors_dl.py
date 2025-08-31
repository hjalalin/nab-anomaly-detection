import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ---------- Models ----------
class LSTMAE(nn.Module):
    def __init__(self, seq_len, n_features, hidden_size=64, num_layers=2, dropout=0.1):
        #super().__init__()
        super(LSTMAE, self).__init__()
        self.seq_len = seq_len
        self.n_features = n_features
        self.embedding_dim = hidden_size

        # Encoder LSTM + LayerNorm
        self.encoder_lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            batch_first=True,
            dropout=dropout,
            num_layers=num_layers 
        )
        self.encoder_norm = nn.LayerNorm(hidden_size)

        # Dropout after encoder
        self.dropout = nn.Dropout(p=dropout)

        # Decoder LSTM
        self.decoder_lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=n_features,
            batch_first=True,
            dropout=dropout,
            num_layers=num_layers
        )

    def forward(self, x):
         # Encoder
        encoded_seq, _ = self.encoder_lstm(x)
        encoded_seq = self.encoder_norm(encoded_seq)

        # Decoder
        decoded_seq, _ = self.decoder_lstm(encoded_seq)
        return decoded_seq


# ---------- Helpers ----------
def _make_loader(X, batch_size, shuffle):
    if X is None or len(X) == 0:
        return None
    if isinstance(X, np.ndarray):
        X_t = torch.from_numpy(X).float()
    else:
        # assume already a torch tensor
        X_t = X.float()
    ds = TensorDataset(X_t)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


# ---------- Training ----------
def train_autoencoder(X_train, X_val, n_features, cfg, verbose=True):
    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMAE(cfg['seq_len'], n_features, cfg["hidden"], cfg["layers"], cfg["dropout"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    loss_fn = nn.MSELoss(reduction="mean")

    train_loader = _make_loader(X_train, cfg["batch_size"], shuffle=True)
    val_loader   = _make_loader(X_val,   cfg["batch_size"], shuffle=False)

    history = {"train_loss": [], "val_loss": []}
    best_val = float("inf")
    best_state = None
    patience_left = cfg["patience"]

    for epoch in range(1, cfg["epochs"] + 1):
        # TRAIN
        model.train(True)
        total = 0.0
        n = 0
        for (batch,) in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(batch)
            loss = loss_fn(pred, batch)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(batch)
            n += len(batch)
        tr = total / max(n, 1)
        history["train_loss"].append(tr)

        # VALIDATION
        if val_loader is not None:
            model.train(False)
            total_v = 0.0
            n_v = 0
            with torch.no_grad():
                for (batch,) in val_loader:
                    batch = batch.to(device)
                    pred = model(batch)
                    loss = loss_fn(pred, batch)
                    total_v += loss.item() * len(batch)
                    n_v += len(batch)
            vl = total_v / max(n_v, 1)
            history["val_loss"].append(vl)

            if verbose:
                print("Epoch {}/{} | train={:.6f} val={:.6f}".format(epoch, cfg["epochs"], tr, vl))

            if vl  < best_val:
                best_val = vl
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                patience_left = cfg["patience"]
            else:
                patience_left -= 1
                if patience_left <= 0:
                    if verbose:
                        print("Early stopping.")
                    break
        else:
            if verbose:
                print("Epoch {}/{} | train={:.6f}".format(epoch, cfg["epochs"], tr))

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history



# -------- reconstruct ---------------
def reconstruction_error(x, x_hat, reduce="last_t", q=0.95):
    """
    x, x_hat: (B, T, F)
    reduce: 'last_t' | 'time_feature_mean' | 'time_feature_max' | 'time_feature_q'
    """
    se = (x - x_hat).pow(2)  # (B, T, F)
    if reduce == "last_t":
        return se[:, -1, :].mean(dim=1)         # (B,)
    if reduce == "time_feature_mean":
        return se.mean(dim=(1, 2))              # (B,)
    if reduce == "time_feature_max":
        return se.amax(dim=(1, 2))              # (B,)
    if reduce == "time_feature_q":
        flat = se.view(se.size(0), -1)          # (B, T*F)
        k = max(1, int(round(q * flat.size(1))))
        vals, _ = torch.topk(flat, k, dim=1)
        return vals[:, -1]                      # (B,)
    raise ValueError(reduce)



def ae_scores(model, data, cfg, reduce="last_t"):
    """
    data: DataLoader OR ndarray/tensor shaped (N, T, F)
    returns: np.ndarray, shape (N,)
    """
    batch_size = cfg["dl"]["autoencoder"]["batch_size"]
    device = cfg["dl"]["autoencoder"]["device"]
    model.eval()
    if isinstance(data, torch.utils.data.DataLoader):
        dl = data
    else:
        if isinstance(data, np.ndarray):
            X = torch.from_numpy(data).float()
        elif isinstance(data, torch.Tensor):
            X = data.float()
        else:
            raise TypeError("data must be DataLoader, np.ndarray, or torch.Tensor")
        dl = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X),
                                         batch_size=batch_size, shuffle=False)

    outs = []
    for (xb,) in dl:
        xb = xb.to(device)
        xh, _ = model.encoder_lstm(xb)            
        xh = model.encoder_norm(xh)
        xh = model.decoder_lstm(xh)[0]
        se = (xb - xh).pow(2)
        if reduce == "last_t":
            s = se[:, -1, :].mean(dim=1)          # (B,)
        elif reduce == "time_feature_mean":
            s = se.mean(dim=(1, 2))
        elif reduce == "time_feature_max":
            s = se.amax(dim=(1, 2))
        else:
            raise ValueError(reduce)
        outs.append(s.detach().cpu())
    return torch.cat(outs).numpy()



def windows_to_pointwise_last(scores, seq_len, total_len, offset=0):
    """
    scores: 1D array (N_windows,)
    Assign each window's score to the last index: offset + i + seq_len - 1
    """
    out = np.full(total_len, np.nan, dtype=np.float32)
    for i, s in enumerate(scores):
        t = offset + i + seq_len - 1
        if 0 <= t < total_len:
            out[t] = float(s)
    return out






