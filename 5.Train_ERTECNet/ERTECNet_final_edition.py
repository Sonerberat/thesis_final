from __future__ import annotations
import argparse
import csv
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple
from torchvision import datasets, transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
from torchvision.datasets.folder import default_loader

try:
    from torchinfo import summary as torchinfo_summary
    _HAS_TORCHINFO = True
except Exception:
    torchinfo_summary = None
    _HAS_TORCHINFO = False

try:
    from torch.utils.tensorboard import SummaryWriter
    _HAS_TENSORBOARD = True
except Exception:
    SummaryWriter = None
    _HAS_TENSORBOARD = False
# Optional plotting deps (only used if --save-cm)
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix
    _HAS_CM_LIBS = True
except Exception:
    _HAS_CM_LIBS = False


# -----------------------------
# Utility
# -----------------------------
def set_seed(seed: int = 42):
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# Enable performance optimizations (PyTorch 2.0+)
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # Optional: enable for Ampere+ GPUs
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_flash_sdp(True)


def esn_parameter_breakdown(esn: nn.Module):
    """Detailed ESN parameter tally for apples-to-apples comparisons."""
    out = {}
    # Core block lists if present
    W_in = getattr(esn, "W_in", None)
    W = getattr(esn, "W", None)
    Wv_in = getattr(esn, "Wv_in", None)
    Wv = getattr(esn, "Wv", None)
    W_out = getattr(esn, "W_out", None)
    b_out = getattr(esn, "b_out", None)

    out["W_in"] = sum(p.numel() for p in (W_in or []))
    out["W"] = sum(p.numel() for p in (W or []))
    out["Wv_in"] = 0 if Wv_in is None else Wv_in.numel()
    out["Wv"] = 0 if Wv is None else Wv.numel()
    out["W_out"] = 0 if W_out is None else W_out.numel()
    out["b_out"] = 0 if b_out is None else b_out.numel()
    out["total"] = sum(out.values())
    return out


# Modern AMP helper (PyTorch ≥ 2.0)
def make_amp(device: torch.device):
    use_amp = (device.type == "cuda")
    def autocast():
        return torch.amp.autocast(
            device_type=device.type,  # 'cuda' or 'cpu'
            enabled=use_amp,
            dtype=torch.float16 if use_amp else torch.bfloat16
        )
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    return autocast, scaler

def ensure_metrics_header(path: Path, header: str):
    
    header_line = header if header.endswith("\n") else header + "\n"
    if not path.exists():
        path.write_text(header_line)
        return
    text = path.read_text()
    if not text.strip():
        path.write_text(header_line)
        return
    if not text.lstrip().startswith(header.split(",")[0]):
        path.write_text(header_line + text)


# -----------------------------
# CNN Feature Extractor: MBConv + ECA + DropPath + SiLU
# -----------------------------
class ECA(nn.Module):
    """Efficient Channel Attention: tiny, no FC, conv over channel dimension."""
    def __init__(self, k: int = 3):
        super().__init__()
        assert k % 2 == 1, "ECA kernel k must be odd"
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        # x: [B, C, H, W]
        y = self.avg(x)                           # [B, C, 1, 1]
        y = y.squeeze(-1).transpose(1, 2)         # [B, 1, C]
        y = self.conv(y)                          # [B, 1, C]
        y = self.sig(y).transpose(1, 2).unsqueeze(-1)  # [B, C, 1, 1]
        return x * y


class SqueezeExcite(nn.Module):
    def __init__(self, c, r=0.25):
        super().__init__()
        hidden = max(8, int(c * r))
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(c, hidden, 1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, c, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        s = self.avg(x)
        s = self.fc(s)
        return x * s


class DropPath(nn.Module):

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x * random_tensor / keep


class MBConv(nn.Module):
    """
    EfficientNet-style MBConv with optional ECA or SE and DropPath.
    - expand: expansion ratio
    - attn: "eca" (default), "se", or None
    - norm: "bn" or "gn"
    """
    def __init__(
        self,
        cin: int,
        cout: int,
        stride: int = 1,
        expand: int = 4,
        attn: str = "eca",
        drop_path: float = 0.0,
        k: int = 3,
        norm: str = "bn",
        se_ratio: float = 0.25,
    ):
        super().__init__()
        assert stride in (1, 2)
        Norm = (lambda c: nn.BatchNorm2d(c)) if norm == "bn" else (lambda c: nn.GroupNorm(32, c))
        use_expand = (expand != 1)
        hidden = cin * expand if use_expand else cin

        layers = []
        # 1x1 expand
        if use_expand:
            layers += [nn.Conv2d(cin, hidden, 1, bias=False), Norm(hidden), nn.SiLU(inplace=True)]
        # depthwise
        layers += [
            nn.Conv2d(hidden, hidden, k, stride=stride, padding=k // 2, groups=hidden, bias=False),
            Norm(hidden), nn.SiLU(inplace=True)
        ]
        # attention
        if attn == "eca":
            layers += [ECA(k=3)]
        elif attn == "se":
            layers += [SqueezeExcite(hidden, r=se_ratio)]

        # 1x1 project
        layers += [nn.Conv2d(hidden, cout, 1, bias=False), Norm(cout)]
        self.block = nn.Sequential(*layers)

        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.has_res = (stride == 1 and cin == cout)
        self.out_act = nn.SiLU(inplace=True)  # lightweight post-act

        # Kaiming init (works well with SiLU/GELU)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        y = self.block(x)
        if self.has_res:
            y = self.drop_path(y) + x
        return self.out_act(y)


class CNNFeatureExtractor(nn.Module):
    """
    Lightweight 3-stage MBConv backbone.
    For small images (≤32), keep stride-1 in the stem to preserve detail.
    """
    def __init__(
        self,
        in_channels: int = 1,
        widths=(32, 64, 128),
        strides=(1, 2, 2),
        expand: int = 4,
        attn: str = "eca",           # change to "se" to use Squeeze-Excite
        drop_path_max: float = 0.1,  # DropPath schedule across depth
        norm: str = "bn",
    ):
        super().__init__()
        w1, w2, w3 = widths
        s1, s2, s3 = strides

        Norm = (lambda c: nn.BatchNorm2d(c)) if norm == "bn" else (lambda c: nn.GroupNorm(32, c))
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, w1, 3, stride=s1, padding=1, bias=False),
            Norm(w1),
            nn.SiLU(inplace=True),
        )

        # linearly increase droppath over depth
        dp = [0.0, drop_path_max * 0.5, drop_path_max]

        self.blocks = nn.Sequential(
            MBConv(w1, w1, stride=1, expand=expand, attn=attn, drop_path=dp[0], k=3, norm=norm),
            MBConv(w1, w2, stride=s2, expand=expand, attn=attn, drop_path=dp[1], k=5, norm=norm),
            MBConv(w2, w2, stride=1,  expand=expand, attn=attn, drop_path=dp[1], k=5, norm=norm),
            MBConv(w2, w3, stride=s3, expand=expand, attn=attn, drop_path=dp[2], k=5, norm=norm),
            MBConv(w3, w3, stride=1,  expand=expand, attn=attn, drop_path=dp[2], k=3, norm=norm),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        return x  # [B, C, H, W]

    @torch.no_grad()
    def feature_shape(self, input_hw: Tuple[int, int]):
        H, W = input_hw
        dev = next(self.parameters()).device
        x = torch.zeros(1, self.stem[0].in_channels, H, W, device=dev)
        training = self.training
        self.eval()
        y = self.forward(x)
        if training:
            self.train()
        return int(y.shape[1]), int(y.shape[2]), int(y.shape[3])  # C_out, H_out, W_out

# -----------------------------
# Reservoir helpers
# -----------------------------
def spectral_radius_scale(W: torch.Tensor, target_rho: float = 0.99):
    with torch.no_grad():
        eigvals = torch.linalg.eigvals(W.detach().cpu()).abs()
        rho = float(eigvals.max().item()) if eigvals.numel() > 0 else 0.0 # handle empty
        if rho > 0:
            W.mul_(target_rho / rho)
    return W

def sparse_mask(shape, sparsity, device):
    return (torch.rand(shape, device=device) < sparsity).float()


@dataclass
class MESNConfig:
    input_dim: int
    num_classes: int
    L: int = 3
    S: int = 4
    neurons_per_deep: int = 20
    sparsity: float = 0.1
    delta: float = 0.9
    ridge_lambda: float = 1e-3
    device: torch.device = None

    @property
    def R(self):
        return self.L * self.neurons_per_deep

    @property
    def theta(self):
        return self.neurons_per_deep // self.S



class MESNReadout(nn.Module):
    def __init__(self, cfg: MESNConfig):
        super().__init__()
        self.cfg = cfg
        L, S, theta = cfg.L, cfg.S, cfg.theta
        device = cfg.device

        self.W_in, self.W = nn.ParameterList(), nn.ParameterList()
        for d in range(L):
            for w in range(S):
                in_dim = cfg.input_dim if d == 0 else (cfg.input_dim + theta)
                Win = torch.empty(theta, in_dim, device=device).uniform_(-1, 1)
                W = torch.empty(theta, theta, device=device).uniform_(-0.5, 0.5)
                W = W * sparse_mask((theta, theta), cfg.sparsity, device)
                spectral_radius_scale(W, 0.99)
                self.W_in.append(nn.Parameter(Win, requires_grad=False))
                self.W.append(nn.Parameter(W, requires_grad=False))

        self.Wv_in = nn.Parameter(torch.empty(L*S, cfg.input_dim, device=device).uniform_(-1, 1), requires_grad=False)
        Wv = torch.empty(L*S, L*S, device=device).uniform_(-0.5, 0.5)
        Wv = Wv * sparse_mask((L*S, L*S), cfg.sparsity, device)
        spectral_radius_scale(Wv, 0.99)
        self.Wv = nn.Parameter(Wv, requires_grad=False)

        self.W_out = nn.Parameter(
            torch.zeros(cfg.R + (L*S), cfg.num_classes, device=device), requires_grad=False
        )
        self.b_out = nn.Parameter(torch.zeros(cfg.num_classes, device=device), requires_grad=False)
        self.w_out_fitted = False

    def _stack_weights(self):
        L, S = self.cfg.L, self.cfg.S
        W_in_layers = []
        W_layers = []
        idx = 0
        for d in range(L):
            layer_in = torch.stack([self.W_in[idx + s] for s in range(S)], dim=0)
            layer_in = layer_in.transpose(1, 2).contiguous()
            layer_W = torch.stack([self.W[idx + s] for s in range(S)], dim=0).contiguous()
            W_in_layers.append(layer_in)
            W_layers.append(layer_W)
            idx += S

        return tuple(W_in_layers), tuple(W_layers)

    def forward_states(self, u):
        """
        Args:
        u: [B, T, input_dim]  (T = H_out*W_out)
        Returns:
        X:  [B, L*S*theta]
        xv: [B, L*S]
        """
        B, T, _ = u.shape
        L, S, theta = self.cfg.L, self.cfg.S, self.cfg.theta
        W_in_layers, W_layers = self._stack_weights()

        states = [u.new_zeros((B, S, theta)) for _ in range(L)]
        xv = u.new_zeros((B, L * S))

        one_minus_delta = 1.0 - self.cfg.delta
        delta = self.cfg.delta

        for t in range(T):
            u_t = u[:, t, :]
            prev_concat = None
            reps = []

            for d in range(L):
                Win = W_in_layers[d]   # [S, in_dim, theta]
                Wrec = W_layers[d]     # [S, theta, theta]
                prev_state = states[d] # [B, S, theta] (now tracks history properly)

                base_input = u_t.unsqueeze(1).expand(-1, S, -1)  # [B, S, C]
                if d == 0:
                    layer_input = base_input                     # [B, S, C]
                else:
                    prev_split = prev_concat.view(B, S, theta)    # [B, S, theta]
                    layer_input = torch.cat([base_input, prev_split], dim=2)  # [B, S, C+theta]

                # [B,S,in] x [S,in,theta] -> [B,S,theta]
                input_term = torch.einsum("bsi,sik->bsk", layer_input, Win)
                # [B,S,theta] x [S,theta,theta] -> [B,S,theta]
                recur_term = torch.einsum("bsi,sij->bsj", prev_state, Wrec)

                x_d = torch.tanh(input_term + recur_term)         # [B, S, theta]

                states[d] = x_d                                   # <- list assignment keeps graph
                reps.append(x_d.mean(dim=2))                       # [B, S]
                prev_concat = x_d.reshape(B, S * theta)            # [B, S*theta]

            x_rep = torch.cat(reps, dim=1)  # [B, L*S]

            xv = one_minus_delta * x_rep + delta * torch.tanh(
                F.linear(u_t, self.Wv_in) + F.linear(xv, self.Wv)
            )

        # Stack final states: [L, B, S, theta] -> [B, L, S, theta]
        state_layers = torch.stack(states, dim=0).permute(1, 0, 2, 3).contiguous()
        X = state_layers.reshape(B, -1)  # [B, R], R = L*S*theta
        return X, xv


    def forward(self, u):
        X, xv = self.forward_states(u)
        feats = torch.cat([X, xv], 1)  # [B, R + L*S]
        if self.w_out_fitted:
            logits = feats @ self.W_out
            if hasattr(self, "b_out") and self.b_out.numel() > 0:
                logits = logits + self.b_out
            return logits
        return torch.zeros(feats.size(0), self.cfg.num_classes, device=feats.device)


class RTECNet(nn.Module):
    def __init__(self, in_channels, num_classes, image_hw, esn_cfg_overrides=None, device=None):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        H, W = image_hw
        # Heuristic widths/strides by input size
        if max(H, W) <= 32:
            widths, strides = (32, 64, 128), (1, 2, 2)
        elif max(H, W) <= 96:
            widths, strides = (48, 96, 192), (2, 2, 2)
        else:
            widths, strides = (64, 128, 256), (2, 2, 2)
        self.cnn = CNNFeatureExtractor(
            in_channels=in_channels,
            widths=widths,
            strides=strides,
            expand=4,
            drop_path_max=0.1,
            norm="bn",
        ).to(self.device)
        C_out, H_out, W_out = self.cnn.feature_shape(image_hw)
        # Spatial tokens: T = H_out * W_out, input_dim = C_out
        T = H_out * W_out
        cfg = MESNConfig(
            input_dim=C_out,
            num_classes=num_classes,
            device=self.device
        )
        # Respect overrides provided by caller
        if esn_cfg_overrides:
            for k, v in esn_cfg_overrides.items():
                setattr(cfg, k, v)
        self.cfg = cfg
        self.esn = MESNReadout(cfg).to(self.device)

    def to_sequence(self, feat_map):  # [B, C, H, W] -> [B, T, C]
        B, C, H, W = feat_map.shape
        return feat_map.permute(0, 2, 3, 1).reshape(B, H * W, C)

    def forward(self, x):
        h = self.cnn(x)                 # [B, C, H, W]
        u_seq = self.to_sequence(h)     # [B, T, C]
        return self.esn(u_seq)


# -----------------------------
# Ridge Solver (robust)
# -----------------------------
class ESNRidgeSolver:
    def __init__(self, feat_dim, num_classes, ridge_lambda, device,
                 use_bias: bool = False, dtype_acc: torch.dtype = torch.float32):
        D = feat_dim + (1 if use_bias else 0)
        self.Sxx = torch.zeros(D, D, device=device, dtype=dtype_acc)
        self.Sxy = torch.zeros(D, num_classes, device=device, dtype=dtype_acc)
        self.lam, self.device, self.use_bias = ridge_lambda, device, use_bias
        self.dtype_acc = dtype_acc
        self._warned_fallback = False

    @torch.no_grad()
    def _with_bias(self, feats: torch.Tensor):
        if not self.use_bias:
            return feats
        ones = torch.ones(feats.size(0), 1, device=feats.device, dtype=feats.dtype)
        return torch.cat([feats, ones], dim=1)

    @torch.no_grad()
    def reset(self):
        self.Sxx.zero_()
        self.Sxy.zero_()

    @torch.no_grad()
    def update(self, feats: torch.Tensor, y: torch.Tensor):
        feats = self._with_bias(feats).to(self.Sxx.dtype)
        Y = F.one_hot(y, self.Sxy.size(1)).to(self.Sxy.dtype)
        self.Sxx.add_(feats.T @ feats)
        self.Sxy.add_(feats.T @ Y)

    @torch.no_grad()
    def solve(self):
        I = torch.eye(self.Sxx.size(0), device=self.device, dtype=self.Sxx.dtype)
        # Data-scaled Tikhonov: keep ridge strength proportional to feature energy
        diag_mean = torch.clamp_min(torch.diag(self.Sxx).mean(), 1e-8)
        lam_eff = self.lam * diag_mean
        A = self.Sxx + lam_eff * I
        A = (A + A.T) * 0.5  # enforce symmetry

        # Try Cholesky with increasing jitter
        for k in range(5):
            jitter = (1e-6 * (10 ** k))
            try:
                L = torch.linalg.cholesky(A + jitter * I)
                W = torch.cholesky_solve(self.Sxy, L)
                return W.to(torch.float32)
            except RuntimeError:
                continue

        # Fallback using eigendecomposition
        if not self._warned_fallback:
            print("[warn] ESNRidgeSolver: Cholesky solve failed after jitter retries; using eigenvalue decomposition.")
            self._warned_fallback = True
        evals, Q = torch.linalg.eigh(A)
        inv = Q @ torch.diag(1.0 / evals.clamp_min(1e-12)) @ Q.T
        W = inv @ self.Sxy
        return W.to(torch.float32)


class KPCAMDataset(torch.utils.data.Dataset):
    """CSV-labeled Kaggle PCam dataset (train_labels.csv + train/*.tif)."""
    def __init__(self, samples, transform=None, class_names=None):
        self.samples = samples
        self.transform = transform
        self.classes = class_names or ["normal", "metastasis"]
        self.loader = default_loader

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = self.loader(path)
        if self.transform:
            img = self.transform(img)
        return img, label


def make_dataloaders(dataset, image_size, batch_size, num_workers,
                     root="./data", dataset_root=None, prefetch_factor=2):
    H, W = image_size
    data_root = dataset_root or root

    def resolve_dataset_dir(base_dir, *required_names):
        base_path = Path(base_dir).expanduser()
        candidates = [base_path, base_path / dataset]
        for candidate in candidates:
            if all((candidate / name).exists() for name in required_names):
                return candidate
        return base_path

    if dataset == "mnist":
        tfm = transforms.Compose([
            transforms.Resize((H, W)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_ds = datasets.MNIST(data_root, train=True, transform=tfm, download=True)
        test_ds  = datasets.MNIST(data_root, train=False, transform=tfm, download=True)
        in_ch, num_classes = 1, 10

    elif dataset == "random_uniklinikum":
        data_dir = dataset_root or os.environ.get("RANDOM_UNIKLINIKUM_DATASET_DIR") or root
        train_dir = os.path.join(data_dir, "train")
        test_dir  = os.path.join(data_dir, "test")
        if not (os.path.isdir(train_dir) and os.path.isdir(test_dir)):
            raise FileNotFoundError(
                f"Expected train/test folders under {data_dir}. "
                f"Set --dataset-root or RANDOM_UNIKLINIKUM_DATASET_DIR if your data lives elsewhere."
            )

        unik_mean = (0.740058, 0.530798, 0.684085)
        unik_std  = (0.181236, 0.226015, 0.167639)

        train_tfm = transforms.Compose([
            transforms.Resize((H, W)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=unik_mean, std=unik_std),
        ])
        test_tfm = transforms.Compose([
            transforms.Resize((H, W)),
            transforms.ToTensor(),
            transforms.Normalize(mean=unik_mean, std=unik_std),
        ])
        train_ds = datasets.ImageFolder(train_dir, transform=train_tfm)
        test_ds  = datasets.ImageFolder(test_dir,  transform=test_tfm)
        if not hasattr(train_ds, "classes") or not train_ds.classes:
            train_ds.classes = ["normal", "metastasis"]
        test_ds.classes = train_ds.classes
        in_ch, num_classes = 3, len(train_ds.classes)

    elif dataset == "k-pcam":
        data_dir = resolve_dataset_dir(dataset_root or root, "train_labels.csv", "train")
        labels_csv = data_dir / "train_labels.csv"
        train_dir = data_dir / "train"
        if not (labels_csv.is_file() and train_dir.is_dir()):
            raise FileNotFoundError(
                f"Expected train_labels.csv and train/ under {data_dir} "
                f"(or under {(Path(dataset_root or root).expanduser() / dataset)}). "
                "Set --dataset-root to the folder containing them."
            )

        samples = []
        with labels_csv.open(newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_id = row.get("id")
                if not img_id:
                    continue
                label = int(row.get("label", 0))
                path = train_dir / f"{img_id}.tif"
                if path.is_file():
                    samples.append((path, label))
        if not samples:
            raise RuntimeError("k-pcam: no training samples found after reading train_labels.csv.")

        train_tfm = transforms.Compose([
            transforms.Resize((H, W)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.02),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.702452, 0.546253, 0.696458),
                                 std=(0.238899, 0.282102, 0.216258)),
        ])
        test_tfm = transforms.Compose([
            transforms.Resize((H, W)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.702452, 0.546253, 0.696458),
                                 std=(0.238899, 0.282102, 0.216258)),
        ])

        rng = np.random.default_rng(42)
        by_label = {}
        for path, label in samples:
            by_label.setdefault(label, []).append((path, label))

        train_samples = []
        test_samples = []
        for items in by_label.values():
            rng.shuffle(items)
            split_idx = int(0.8 * len(items))
            train_samples.extend(items[:split_idx])
            test_samples.extend(items[split_idx:])
        rng.shuffle(train_samples)
        rng.shuffle(test_samples)

        train_ds = KPCAMDataset(train_samples, transform=train_tfm)
        test_ds = KPCAMDataset(test_samples, transform=test_tfm, class_names=train_ds.classes)
        in_ch, num_classes = 3, len(train_ds.classes)

    else:
        raise ValueError("Unsupported dataset")

    loader_kwargs = {
        "pin_memory": True,
        "persistent_workers": num_workers > 0,
        "num_workers": num_workers,
    }
    if num_workers > 0 and prefetch_factor:
        loader_kwargs["prefetch_factor"] = prefetch_factor

    train_loader = DataLoader(
        train_ds, batch_size, shuffle=True,
        **loader_kwargs
    )
    test_loader = DataLoader(
        test_ds, batch_size, shuffle=False,
        **loader_kwargs
    )

    return train_loader, test_loader, in_ch, num_classes


@torch.no_grad()
def evaluate(
    model,
    loader,
    save_cm_path: str = None,
    save_roc_path: str = None,
    class_names=None,
    compute_roc_auc: bool = False,
):
    """
    Evaluate without keeping every prediction/probability in memory.

    Metrics (acc/precision/recall/F1) are derived from a streaming confusion matrix.
    Probabilities are only collected if ROC/AUC is requested to avoid large RAM spikes.
    """
    model.eval()
    correct = total = 0
    total_loss = 0.0
    num_classes = None
    conf_mat = None
    criterion = nn.CrossEntropyLoss()

    need_probs = compute_roc_auc or bool(save_roc_path)
    prob_chunks = [] if need_probs else None
    label_chunks = [] if need_probs else None

    for x, y in loader:
        x = x.to(model.device, non_blocking=True)
        if model.device.type == 'cuda':
            x = x.to(memory_format=torch.channels_last)
        y = y.to(model.device, non_blocking=True)

        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        pred = logits.argmax(1)

        if num_classes is None:
            num_classes = logits.shape[1]
            conf_mat = torch.zeros((num_classes, num_classes), dtype=torch.int64)

        y_cpu = y.detach().cpu()
        pred_cpu = pred.detach().cpu()

        correct += (pred_cpu == y_cpu).sum().item()
        total += y_cpu.numel()
        total_loss += criterion(logits, y).item() * y.size(0)

        # Stream confusion counts: row=true, col=pred
        bincount = torch.bincount(
            y_cpu * num_classes + pred_cpu,
            minlength=num_classes * num_classes,
        ).reshape(num_classes, num_classes)
        conf_mat += bincount

        if need_probs:
            prob_chunks.append(probs.detach().cpu().numpy())
            label_chunks.append(y_cpu.numpy())

    avg_loss = total_loss / total if total > 0 else 0.0

    # Macro precision/recall/F1 from confusion matrix (mirrors sklearn with zero_division=0)
    precision = recall = f1 = 0.0
    if conf_mat is not None:
        tp = torch.diag(conf_mat).double()
        fp = conf_mat.sum(0).double() - tp
        fn = conf_mat.sum(1).double() - tp
        present = ((tp + fn) > 0) | ((tp + fp) > 0)  # classes seen in y_true or y_pred

        if present.any():
            precision_per = torch.zeros_like(tp)
            recall_per = torch.zeros_like(tp)
            f1_per = torch.zeros_like(tp)

            prec_denom = tp + fp
            prec_mask = prec_denom > 0
            precision_per[prec_mask] = tp[prec_mask] / prec_denom[prec_mask]

            rec_denom = tp + fn
            rec_mask = rec_denom > 0
            recall_per[rec_mask] = tp[rec_mask] / rec_denom[rec_mask]

            f1_denom = 2 * tp + fp + fn
            f1_mask = f1_denom > 0
            f1_per[f1_mask] = 2 * tp[f1_mask] / f1_denom[f1_mask]

            precision = precision_per[present].mean().item()
            recall = recall_per[present].mean().item()
            f1 = f1_per[present].mean().item()

    roc_auc = None
    if need_probs and label_chunks:
        probs_arr = np.concatenate(prob_chunks, axis=0)
        labels_arr = np.concatenate(label_chunks, axis=0)
        if np.unique(labels_arr).size > 1:
            try:
                if probs_arr.shape[1] == 2:
                    roc_auc = roc_auc_score(labels_arr, probs_arr[:, 1])
                else:
                    roc_auc = roc_auc_score(labels_arr, probs_arr, multi_class="ovr", average="macro")
            except Exception as exc:
                print(f"[warn] ROC AUC computation failed: {exc}")
                roc_auc = None
        elif compute_roc_auc:
            print("[warn] ROC AUC requested but only one class present; skipping.")

    if save_cm_path and _HAS_CM_LIBS and conf_mat is not None:
        Path(save_cm_path).parent.mkdir(parents=True, exist_ok=True)
        cm = conf_mat.numpy().astype(np.float64)
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums > 0)
        plt.figure(figsize=(7, 6))
        sns.heatmap(
            cm_norm, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=class_names if class_names else range(cm.shape[0]),
            yticklabels=class_names if class_names else range(cm.shape[0])
        )
        plt.ylabel("True label")
        plt.xlabel("Predicted label")
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.savefig(save_cm_path, dpi=200)
        plt.close()
    elif save_cm_path and not _HAS_CM_LIBS:
        print("[warn] --save-cm requested but matplotlib/seaborn not available; skipping CM image.")

    if save_roc_path:
        if not _HAS_CM_LIBS:
            print("[warn] --save-roc requested but matplotlib/seaborn not available; skipping ROC image.")
        elif not need_probs or not label_chunks or np.unique(np.concatenate(label_chunks)).size <= 1:
            print("[warn] --save-roc requested but ROC curve is undefined (need at least two classes).")
        else:
            probs_arr = np.concatenate(prob_chunks, axis=0)
            labels_arr = np.concatenate(label_chunks, axis=0)
            num_classes = probs_arr.shape[1]
            try:
                Path(save_roc_path).parent.mkdir(parents=True, exist_ok=True)
                plt.figure(figsize=(7, 6))
                plt.plot([0, 1], [0, 1], "k--", label="Chance")
                if num_classes == 2:
                    fpr, tpr, _ = roc_curve(labels_arr, probs_arr[:, 1])
                    auc_val = auc(fpr, tpr)
                    roc_label = f"ROC (AUC={auc_val:.3f})"
                    plt.plot(fpr, tpr, label=roc_label)
                else:
                    classes = class_names if class_names is not None else list(range(num_classes))
                    y_bin = label_binarize(labels_arr, classes=list(range(num_classes)))
                    for idx in range(num_classes):
                        # Skip if class has no positive or negative samples
                        if y_bin[:, idx].sum() == 0 or y_bin[:, idx].sum() == y_bin.shape[0]:
                            continue
                        fpr, tpr, _ = roc_curve(y_bin[:, idx], probs_arr[:, idx])
                        auc_val = auc(fpr, tpr)
                        lbl = str(classes[idx])
                        plt.plot(fpr, tpr, label=f"{lbl} (AUC={auc_val:.3f})")
                plt.xlabel("False Positive Rate")
                plt.ylabel("True Positive Rate")
                plt.title("ROC Curve")
                plt.legend(loc="lower right")
                plt.tight_layout()
                plt.savefig(save_roc_path, dpi=200)
                plt.close()
            except Exception as exc:
                print(f"[warn] ROC plotting failed: {exc}")

    return (correct / total) if total > 0 else 0.0, f1, precision, recall, avg_loss, roc_auc


# -----------------------------
# Warm start for W_out
# -----------------------------
@torch.no_grad()
def warm_start_wout(model, loader, ridge_lambda, num_classes, max_batches=None):
    """Initialize W_out before training so CNN gradients are meaningful from epoch 1."""
    feat_dim = model.cfg.R + (model.cfg.L * model.cfg.S)
    solver = ESNRidgeSolver(feat_dim, num_classes, ridge_lambda, model.device,
                            use_bias=False, dtype_acc=torch.float64)
    processed = 0
    for b, (x, y) in enumerate(loader):
        if max_batches is not None and b >= max_batches:
            break
        x = x.to(model.device, non_blocking=True)
        if model.device.type == 'cuda':
            x = x.to(memory_format=torch.channels_last)
        y = y.to(model.device, non_blocking=True)
        # forward to spatial tokens
        feat_map = model.cnn(x)
        u_seq = model.to_sequence(feat_map)
        X, xv = model.esn.forward_states(u_seq)
        Z = torch.cat([X, xv], 1)
        Z = torch.nan_to_num(Z, nan=0.0, posinf=1e4, neginf=-1e4)
        solver.update(Z, y)
        processed += x.size(0)

    if processed > 0:
        W_full = solver.solve()  # [D(+1), C]
        if solver.use_bias:
            model.esn.W_out.copy_(W_full[:-1, :])
            model.esn.b_out.copy_(W_full[-1, :])
        else:
            model.esn.W_out.copy_(W_full)
        model.esn.w_out_fitted = True


# -----------------------------
# Main
# -----------------------------
def main():
    start_time = time.time()
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="mnist",
                   choices=["mnist", "random_uniklinikum", "k-pcam"])
    p.add_argument("--dataset-root", type=str, default="/home/erensr/data",
                   help="Root path for dataset files. For random_uniklinikum set to the dataset folder (train/test subdirs). Can also set RANDOM_UNIKLINIKUM_DATASET_DIR.")
    p.add_argument("--image-size", type=int, nargs=2, default=[28, 28])
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--weight-decay", type=float, default=0.00005,
                   help="AdamW weight decay applied to CNN parameters.")
    p.add_argument("--neurons-per-deep", type=int, default=20, help="Number of neurons per deep reservoir layer.")
    p.add_argument("--sub-reservoirs", type=int, default=4, help="Number of sub-reservoirs (S) per layer.")
    p.add_argument("--warmup-batches", type=int, default=1, help="Batches to warm-start W_out (0 uses the full train set).")

    p.add_argument("--torchinfo", action="store_true", help="Print a torchinfo summary of the model before training.")
    p.add_argument("--tensorboard-logdir", type=str, default=None, metavar="DIR",
                   help="Write a TensorBoard graph of the CNN backbone to DIR.")

    # Optional confusion-matrix saving
    p.add_argument("--save-cm", action="store_true", help="Save confusion-matrix image after evaluation.")
    p.add_argument("--cm-dir", type=str, default="/home/erensr/ERTECnet/unioutput/cm", help="Directory to save confusion matrices.")
    p.add_argument("--cm-every", type=int, default=1, help="Save a CM every N epochs (default 1).")
    # Optional ROC saving
    p.add_argument("--compute-roc-auc", action="store_true",
                   help="Compute ROC AUC during eval (collects probabilities; uses more memory). "
                        "Automatically enabled when --save-roc is used.")
    p.add_argument("--save-roc", action="store_true", help="Save ROC curve image after evaluation.")
    p.add_argument("--roc-dir", type=str, default="/home/erensr/ERTECnet/unioutput/roc", help="Directory to save ROC curves.")
    p.add_argument("--roc-every", type=int, default=1, help="Save a ROC curve every N epochs (default 1).")
    # Optional metrics logging
    p.add_argument("--metrics-path", type=str, default="",
                   help="CSV file to append per-epoch metrics. Use empty string to disable.")
    p.add_argument("--save-best-path", type=str, default="",
                   help="Path to save best-performing checkpoint (by test accuracy). Only saves if provided.")

    args = p.parse_args()

    # Dataset root resolution (with convenience defaults/env override for random_uniklinikum)
    if args.dataset == "random_uniklinikum":
        env_unik = os.environ.get("RANDOM_UNIKLINIKUM_DATASET_DIR")
        if env_unik:
            dataset_root = Path(env_unik).expanduser()
        else:
            default_unik = Path("/home/erensr/data/random_uniklinikum")
            if args.dataset_root == "./data" and default_unik.exists():
                dataset_root = default_unik
            else:
                dataset_root = Path(args.dataset_root).expanduser()
    else:
        dataset_root = Path(args.dataset_root).expanduser()
    dataset_root = dataset_root.resolve()

    # Normalize checkpoint path
    save_best_path = args.save_best_path
    if save_best_path is not None and save_best_path.strip() == "":
        save_best_path = None
    if save_best_path:
        save_best_path = Path(save_best_path)
        save_best_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_header = "epoch,train_loss,test_loss,test_acc,f1,precision,recall,roc_auc,cm_path,roc_path,epoch_time_sec"
    metrics_path = args.metrics_path
    if metrics_path is not None and metrics_path.strip() == "":
        metrics_path = None
    if metrics_path:
        metrics_path = Path(metrics_path)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        ensure_metrics_header(metrics_path, metrics_header)
        print(f"[metrics] Logging to {metrics_path} with columns: {metrics_header}")

    # Validate divisibility
    if args.neurons_per_deep % args.sub_reservoirs != 0:
        raise ValueError(
            f"--neurons-per-deep ({args.neurons_per_deep}) must be divisible by "
            f"--sub-reservoirs ({args.sub_reservoirs})"
        )

    # Repro & device
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    print(f"\n{'='*50}")
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"cuDNN Version: {torch.backends.cudnn.version()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print(f"{'='*50}\n")

    if args.save_cm:
        os.makedirs(args.cm_dir, exist_ok=True)
    if args.save_roc:
        os.makedirs(args.roc_dir, exist_ok=True)

    # Data
    worker_target = os.cpu_count() or 2
    num_workers = max(2, min(8, worker_target // 2))
    data_root_str = dataset_root.as_posix()
    train_loader, test_loader, in_ch, num_classes = make_dataloaders(
        args.dataset, tuple(args.image_size), args.batch_size,
        num_workers=num_workers, root=data_root_str, dataset_root=data_root_str,
        prefetch_factor=4
    )
    class_names = getattr(getattr(train_loader, "dataset", None), "classes", list(range(num_classes)))

    # Model
    esn_cfg_overrides = {
        "neurons_per_deep": args.neurons_per_deep,
        "S": args.sub_reservoirs,
    }

    model = RTECNet(
        in_ch, num_classes, tuple(args.image_size),
        esn_cfg_overrides=esn_cfg_overrides,
        device=device
    )
    checkpoint_target = model

    if args.torchinfo:
        if not _HAS_TORCHINFO:
            print("[warn] torchinfo is not installed. Run 'pip install torchinfo' to enable summaries.")
        else:
            try:
                print("\nTorchinfo Summary (RTECNet)")
                dummy = torch.zeros((1, in_ch, *args.image_size), device=getattr(model, "device", device))
                was_training = model.training
                model.eval()
                with torch.no_grad():
                    torchinfo_summary(
                        model,
                        input_data=dummy,
                        col_names=("input_size", "output_size", "num_params"),
                        depth=4,
                    )
                if was_training:
                    model.train()
            except Exception as exc:
                print(f"[warn] torchinfo summary failed: {exc}")

    if args.tensorboard_logdir:
        if not _HAS_TENSORBOARD:
            print("[warn] TensorBoard is not installed. Run 'pip install tensorboard' to enable logging.")
        else:
            logdir = Path(args.tensorboard_logdir)
            logdir.mkdir(parents=True, exist_ok=True)
            writer = SummaryWriter(log_dir=logdir.as_posix())
            dummy = torch.randn((1, in_ch, *args.image_size), device=getattr(model, "device", device))
            was_training = model.training
            was_cnn_training = model.cnn.training
            model.eval()
            model.cnn.eval()
            try:
                with torch.no_grad():
                    writer.add_graph(model.cnn, dummy)
                print(f"[tensorboard] Wrote CNN graph to {logdir}")
            except Exception as exc:
                print(f"[warn] TensorBoard logging failed: {exc}")
            finally:
                writer.close()
                if was_cnn_training:
                    model.cnn.train()
                if was_training:
                    model.train()

    # Channels-last layout for potential speedups
    if device.type == 'cuda':
        model = model.to(memory_format=torch.channels_last)
        checkpoint_target = model

    if hasattr(torch, "compile"):
        try:
            model = torch.compile(model)
            checkpoint_target = getattr(model, "_orig_mod", model)
        except Exception as exc:
            print(f"[warn] torch.compile unavailable ({exc}). Continuing without it.")
            checkpoint_target = model

    # ESN detailed breakdown 
    esn_break = esn_parameter_breakdown(model.esn)
    print("ESN Parameter Breakdown")
    print("-"*50)
    for k in ("W_in", "W", "Wv_in", "Wv", "W_out", "b_out"):
        print(f"{k:<10}: {esn_break[k]:,}")
    print("-"*50)
    print(f"ESN total: {esn_break['total']:,}")
    print("="*50 + "\n")

    #  Log ESN config
    print(f"ESN Config: L={model.cfg.L}, S={model.cfg.S}, "
          f"neurons_per_deep={model.cfg.neurons_per_deep}, "
          f"theta={model.cfg.theta}, R={model.cfg.R}")
    print(f"Model CNN is on: {next(model.cnn.parameters()).device}")
    print(f"Model ESN is on: {next(model.esn.parameters()).device}")

    optimizer = torch.optim.AdamW(
        [{"params": model.cnn.parameters(), "weight_decay": args.weight_decay}],
        lr=args.lr,
        amsgrad=True,
        weight_decay=args.weight_decay,
    )

    # AMP (modern)
    autocast, scaler = make_amp(device)

    # Warm-start W_out so gradients flow from epoch 1
    warm_batches = None if args.warmup_batches == 0 else args.warmup_batches
    scope = "full train set" if warm_batches is None else f"first {warm_batches} batches"
    print(f"Warm-starting W_out using the {scope}...")
    warm_start_wout(model, train_loader, model.cfg.ridge_lambda, num_classes, max_batches=warm_batches)

    # Training loop follows...
    feat_dim = model.cfg.R + (model.cfg.L * model.cfg.S)
    solver = ESNRidgeSolver(feat_dim, num_classes, model.cfg.ridge_lambda, device,
                            use_bias=False, dtype_acc=torch.float64)
    best_acc = -1.0
    best_epoch = -1

    # Training
    for epoch in range(args.epochs):
        epoch_start = time.time()
        model.train()
        total_loss, num_batches = 0.0, 0

        # Stream ridge stats over this epoch
        solver.reset()

        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            if device.type == 'cuda':
                x = x.to(memory_format=torch.channels_last)
            y = y.to(device, non_blocking=True)

            # 1) CNN under AMP (fast)
            with autocast():
                feat_map = model.cnn(x)  # [B, C, H, W]

            # 2) ESN + ridge features (FP32 for stability)
            u_seq = model.to_sequence(feat_map)         # [B, T, C]
            X, xv = model.esn.forward_states(u_seq.float())
            Z = torch.cat([X, xv], 1)
            if not torch.isfinite(Z).all():
                Z = torch.nan_to_num(Z, nan=0.0, posinf=1e4, neginf=-1e4)

            # Update ridge stats with detached features
            solver.update(Z.detach(), y)

            # 3) Train CNN using current readout
            logits = Z @ model.esn.W_out
            if hasattr(model.esn, "b_out"):
                logits = logits + model.esn.b_out
            loss = F.cross_entropy(logits, y,label_smoothing=0.02)

            optimizer.zero_grad(set_to_none=True)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.cnn.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.cnn.parameters(), max_norm=5.0)
                optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        # Fit W_out at end of epoch (robust solve)
        W_full = solver.solve()  # [D(+1), C]
        if solver.use_bias:
            model.esn.W_out.copy_(W_full[:-1, :])
            model.esn.b_out.copy_(W_full[-1, :])
        else:
            model.esn.W_out.copy_(W_full)
        model.esn.w_out_fitted = True

        # Evaluate (+ optional CM)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        cm_path = None
        if args.save_cm and ((epoch + 1) % args.cm_every == 0):
            cm_path = os.path.join(args.cm_dir, f"confusion_matrix_epoch{epoch+1}.png")
        roc_path = None
        if args.save_roc and ((epoch + 1) % args.roc_every == 0):
            roc_path = os.path.join(args.roc_dir, f"roc_epoch{epoch+1}.png")

        want_roc_metrics = args.compute_roc_auc or args.save_roc
        test_acc, f1, precision, recall, test_loss, roc_auc = evaluate(
            model,
            test_loader,
            save_cm_path=cm_path,
            save_roc_path=roc_path,
            class_names=class_names,
            compute_roc_auc=want_roc_metrics,
        )
        epoch_time = time.time() - epoch_start
        roc_display = f"{roc_auc:.4f}" if roc_auc is not None else "n/a"
        if metrics_path:
            roc_field = "" if roc_auc is None else f"{roc_auc:.6f}"
            with metrics_path.open("a") as f:
                f.write(
                    f"{epoch+1},{total_loss/num_batches:.6f},{test_loss:.6f},"
                    f"{test_acc:.6f},{f1:.6f},{precision:.6f},{recall:.6f},"
                    f"{roc_field},{cm_path or ''},{roc_path or ''},"
                    f"{epoch_time:.3f}\n"
                )
        msg = (
            f"Epoch {epoch+1}/{args.epochs}: "
            f"Train Loss = {total_loss/num_batches:.4f} | "
            f"Test Loss = {test_loss:.4f} | "
            f"Test Acc = {test_acc*100:.2f}% | "
            f"F1 = {f1:.4f} | Prec = {precision:.4f} | Recall = {recall:.4f} | "
            f"ROC AUC = {roc_display} | "
            f"Time = {epoch_time:.2f}s"
        )
        if cm_path:
            msg += f" | Saved CM: {cm_path}"
        if roc_path:
            msg += f" | Saved ROC: {roc_path}"
        print(msg)
        if save_best_path and test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch + 1
            payload = {
                "epoch": best_epoch,
                "state_dict": checkpoint_target.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "args": vars(args),
                "metrics": {
                    "test_acc": test_acc,
                    "test_loss": test_loss,
                    "f1": f1,
                    "precision": precision,
                    "recall": recall,
                    "roc_auc": roc_auc,
                },
                "esn_cfg": vars(checkpoint_target.cfg),
            }
            torch.save(payload, save_best_path)
            print(f"[checkpoint] Saved new best model to {save_best_path} (acc={test_acc*100:.2f}%)")

    total_time = time.time() - start_time
    print(f"\n{'='*50}")
    print(f"Total execution time: {total_time:.2f} seconds")
    print(f"Average time per epoch: {total_time/args.epochs:.2f} seconds")
    print(f"{'='*50}")
    if save_best_path and best_epoch != -1:
        print(f"Best checkpoint: epoch {best_epoch}, acc={best_acc*100:.2f}% -> {save_best_path}")


if __name__ == "__main__":
    main()
