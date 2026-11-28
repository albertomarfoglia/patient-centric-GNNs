"""Full-batch training with early stopping, and prediction."""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .config import TrainConfig

log = logging.getLogger(__name__)


@dataclass
class TrainResult:
    epochs: int
    best_epoch: int
    best_val_loss: float


def _loss(out: torch.Tensor, idx: torch.Tensor, y: torch.Tensor, binary: bool) -> torch.Tensor:
    if binary:
        return F.binary_cross_entropy_with_logits(out[idx], y.float())
    return F.cross_entropy(out[idx], y)


def train(model, data, train_idx, train_y, val_idx, val_y, cfg: TrainConfig, binary: bool) -> TrainResult:
    """Adam; stop when the validation loss has not improved by ``min_delta``
    for ``patience`` epochs; restore the best weights (kept in memory)."""
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    best_loss, best_epoch, best_state, waited = float("inf"), 0, None, 0
    epoch = 0
    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        opt.zero_grad()
        loss = _loss(model(data), train_idx, train_y, binary)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            val_loss = float(_loss(model(data), val_idx, val_y, binary))
        if val_loss + cfg.min_delta < best_loss:
            best_loss, best_epoch, waited = val_loss, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            waited += 1
            if waited >= cfg.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    log.debug("stopped at epoch %d (best %d, val loss %.4f)", epoch, best_epoch, best_loss)
    return TrainResult(epochs=epoch, best_epoch=best_epoch, best_val_loss=best_loss)


@torch.no_grad()
def predict_proba(model, data, idx: torch.Tensor, binary: bool) -> torch.Tensor:
    """Class probabilities, shape (len(idx), n_classes)."""
    model.eval()
    out = model(data)[idx]
    if binary:
        p = torch.sigmoid(out)
        return torch.stack([1 - p, p], dim=1)
    return torch.softmax(out, dim=1)
