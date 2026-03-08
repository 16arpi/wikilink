#!/usr/bin/env python3
"""
optimize_hyperparams.py – Optuna‑driven hyperparameter search for
CamemBERT‑v2 NER fine‑tuning (IOB2, 3 labels).

Usage
-----
    python optimize_hyperparams.py \
        --dataset ./prepared_data \
        --n-trials 30 \
        --seed 42 \
        --trials-dir ./optuna_results \
        --subsample-frac 0.2
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import logging
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import optuna
import torch
import torch.nn as nn
from datasets import DatasetDict, load_from_disk
from seqeval.metrics import f1_score as seqeval_f1
from torch.cuda.amp import GradScaler, autocast
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModel, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup

# ──────────────────────────────────────────────────────────────────────
# Optional CRF import
# ──────────────────────────────────────────────────────────────────────
CRF_AVAILABLE = False
try:
    from torchcrf import CRF

    CRF_AVAILABLE = True
    logging.getLogger(__name__).info("torchcrf available – CRF head enabled.")
except ImportError:
    try:
        from pytorch_crf import CRF  # type: ignore[no-redef]

        CRF_AVAILABLE = True
    except ImportError:
        pass

# ──────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────
NUM_LABELS = 3
ID2LABEL = {0: "O", 1: "B-ENT", 2: "I-ENT"}
PAD_TOKEN_ID = 0  # CamemBERT‑v2 pad token id
PAD_LABEL_ID = -100  # standard ignore_index for CrossEntropyLoss
MAX_GPU_BATCH = 24  # aggressive AMP mode for T4 16 GB @ seq512
NUM_EPOCHS = 5  # per Optuna trial
MODEL_NAME = "almanach/camembertv2-base"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# Silence chatty loggers
optuna.logging.set_verbosity(optuna.logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)


# ======================================================================
#                           NER MODEL
# ======================================================================
class NERModel(nn.Module):
    """CamemBERT encoder + configurable classification head.

    Supported ``head_type`` values:
      * ``"linear"`` — single projection
      * ``"mlp"``    — multi‑layer perceptron (configurable depth / width)
      * ``"crf"``    — linear emission layer + CRF (requires ``torchcrf``)
    """

    def __init__(
        self,
        encoder: nn.Module,
        head_type: str,
        num_labels: int = NUM_LABELS,
        dropout: float = 0.1,
        # MLP‑specific
        mlp_layers: int = 1,
        mlp_hidden: int = 512,
        activation: str = "gelu",
    ):
        super().__init__()
        self.encoder = encoder
        self.head_type = head_type
        self.num_labels = num_labels
        hidden_size = encoder.config.hidden_size

        self.drop = nn.Dropout(dropout)

        # ── Build the classification head ────────────────────────────
        if head_type == "linear":
            self.classifier = nn.Linear(hidden_size, num_labels)

        elif head_type == "mlp":
            act_fn = nn.GELU() if activation == "gelu" else nn.ReLU()
            layers: list[nn.Module] = []
            in_dim = hidden_size
            for _ in range(mlp_layers):
                layers.extend([nn.Linear(in_dim, mlp_hidden), act_fn, nn.Dropout(dropout)])
                in_dim = mlp_hidden
            layers.append(nn.Linear(in_dim, num_labels))
            self.classifier = nn.Sequential(*layers)

        elif head_type == "crf":
            if not CRF_AVAILABLE:
                raise RuntimeError(
                    "CRF head requested but torchcrf / pytorch-crf is not installed."
                )
            self.emission = nn.Linear(hidden_size, num_labels)
            self.crf = CRF(num_labels, batch_first=True)

        else:
            raise ValueError(f"Unknown head_type: {head_type!r}")

    # -----------------------------------------------------------------
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Returns a dict with keys:
          * ``loss``   – scalar (only when ``labels`` is provided)
          * ``logits`` – (B, T, C) emission logits (non‑CRF heads)
          * ``decoded``– list[list[int]] Viterbi paths (CRF head, inference)
        """
        # Encoder forward  — autocast handles precision externally
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        hidden = self.drop(hidden)

        out: Dict[str, Any] = {}

        if self.head_type == "crf":
            emissions = self.emission(hidden)  # (B, T, C)
            mask = attention_mask.bool()
            out["logits"] = emissions
            if labels is not None:
                # CRF negative log‑likelihood (labels must be ≥0, padding
                # handled via mask; see collator clamping).
                # NOTE: PyTorch autocast automatically keeps numerically
                # sensitive operations (logsumexp) in FP32. We do not
                # disable autocast for the CRF.
                loss = -self.crf(emissions, labels, mask=mask, reduction="mean")
                out["loss"] = loss
            else:
                out["decoded"] = self.crf.decode(emissions, mask=mask)
        else:
            logits = self.classifier(hidden)  # (B, T, C)
            out["logits"] = logits
            if labels is not None:
                loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_LABEL_ID)
                loss = loss_fn(logits.view(-1, self.num_labels), labels.view(-1))
                out["loss"] = loss

        return out


# ======================================================================
#                          DATA COLLATOR
# ======================================================================
def ner_collate_fn(
    batch: List[Dict[str, Any]],
    pad_token_id: int = PAD_TOKEN_ID,
    crf_mode: bool = False,
) -> Dict[str, Any]:
    """Pad variable‑length sequences; keep ``offset_mapping`` unpadded.

    For CRF mode, labels are clamped to 0 instead of -100 because
    ``torchcrf`` requires all tag indices ∈ [0, num_tags). The CRF
    mask (``attention_mask.bool()``) prevents these dummy values from
    contributing to the loss.
    """
    input_ids = [torch.tensor(ex["input_ids"], dtype=torch.long) for ex in batch]
    attention = [torch.tensor(ex["attention_mask"], dtype=torch.long) for ex in batch]
    labels = [torch.tensor(ex["labels"], dtype=torch.long) for ex in batch]
    offsets = [ex["offset_mapping"] for ex in batch]  # list[list[list[int]]] — NOT padded

    input_ids_pad = pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
    attention_pad = pad_sequence(attention, batch_first=True, padding_value=0)
    labels_pad = pad_sequence(labels, batch_first=True, padding_value=PAD_LABEL_ID)

    if crf_mode:
        labels_pad = torch.clamp(labels_pad, min=0)

    return {
        "input_ids": input_ids_pad,
        "attention_mask": attention_pad,
        "labels": labels_pad,
        "offset_mapping": offsets,
    }


# ======================================================================
#                       METRIC COMPUTATION
# ======================================================================
def tokens_to_entities(
    pred_logits: Optional[Any],
    label_ids: Any,
    offset_mapping_batch: List[List[List[int]]],
    id2label: Dict[int, str],
    crf_decoded: Optional[List[List[int]]] = None,
) -> Tuple[List[List[str]], List[List[str]]]:
    """Convert token‑level logits / labels → word‑level IOB2 sequences.

    * Special tokens (offset ``[0, 0]``) are skipped.
    * Only the first sub‑token of each word is retained (word boundary
      when ``tok_start > prev_end``).
    * For CRF, ``crf_decoded`` already contains Viterbi paths
      (variable‑length); ``zip`` truncates to the shortest sequence,
      which is correct by construction.
    """
    if crf_decoded is not None:
        pred_ids_batch = crf_decoded
    else:
        if hasattr(pred_logits, "detach"):
            pred_logits = pred_logits.detach().cpu().numpy()
        pred_ids_batch = np.argmax(pred_logits, axis=-1).tolist()

    if hasattr(label_ids, "detach"):
        label_ids = label_ids.detach().cpu().numpy()
    label_ids_batch = (
        label_ids.tolist() if hasattr(label_ids, "tolist") else list(label_ids)
    )

    all_true: List[List[str]] = []
    all_pred: List[List[str]] = []

    for pred_ids, true_ids, offsets in zip(
        pred_ids_batch, label_ids_batch, offset_mapping_batch
    ):
        pred_seq: List[str] = []
        true_seq: List[str] = []
        prev_end = -1

        for pred_id, true_id, offset in zip(pred_ids, true_ids, offsets):
            if isinstance(offset, dict):
                tok_start, tok_end = int(offset["start"]), int(offset["end"])
            else:
                tok_start, tok_end = int(offset[0]), int(offset[1])

            # Skip special tokens (CLS, SEP, PAD)
            if tok_start == 0 and tok_end == 0:
                continue

            # Word‑boundary detection: first sub‑token only
            is_first_subtoken = tok_start > prev_end
            prev_end = tok_end

            if not is_first_subtoken:
                continue

            # Skip padding labels (-100)
            if true_id == PAD_LABEL_ID:
                continue

            pred_seq.append(id2label.get(pred_id, "O"))
            true_seq.append(id2label.get(true_id, "O"))

        all_true.append(true_seq)
        all_pred.append(pred_seq)

    return all_true, all_pred


def compute_entity_macro_f1(
    pred_logits: Optional[Any],
    label_ids: Any,
    offset_mapping_batch: List[List[List[int]]],
    id2label: Dict[int, str] = ID2LABEL,
    crf_decoded: Optional[List[List[int]]] = None,
) -> float:
    """Entity‑level macro F1 via ``seqeval``."""
    true_seqs, pred_seqs = tokens_to_entities(
        pred_logits=pred_logits,
        label_ids=label_ids,
        offset_mapping_batch=offset_mapping_batch,
        id2label=id2label,
        crf_decoded=crf_decoded,
    )
    if not any(true_seqs):
        return 0.0
    return seqeval_f1(true_seqs, pred_seqs, average="macro", zero_division=0)


# ======================================================================
#                          EVALUATION
# ======================================================================
@torch.no_grad()
def evaluate(
    model: NERModel,
    dataloader: DataLoader,
    device: torch.device,
) -> float:
    """Run evaluation and return entity‑level macro F1."""
    model.eval()

    all_pred_logits: List[torch.Tensor] = []
    all_label_ids: List[torch.Tensor] = []
    all_offsets: List[List[List[int]]] = []
    all_crf_decoded: Optional[List[List[int]]] = None
    is_crf = model.head_type == "crf"

    if is_crf:
        all_crf_decoded = []

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attn_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        offsets = batch["offset_mapping"]

        with autocast():
            out = model(input_ids, attn_mask)

        if is_crf:
            assert all_crf_decoded is not None
            all_crf_decoded.extend(out["decoded"])  # list[list[int]]
        else:
            all_pred_logits.append(out["logits"].cpu())

        all_label_ids.append(labels.cpu())
        all_offsets.extend(offsets)

    if is_crf:
        # CRF: decoded paths are variable‑length lists; labels padded
        # zip truncation in tokens_to_entities handles alignment.
        labels_cat = torch.cat(all_label_ids, dim=0)
        f1 = compute_entity_macro_f1(
            pred_logits=None,
            label_ids=labels_cat,
            offset_mapping_batch=all_offsets,
            crf_decoded=all_crf_decoded,
        )
    else:
        logits_cat = torch.cat(all_pred_logits, dim=0)
        labels_cat = torch.cat(all_label_ids, dim=0)
        f1 = compute_entity_macro_f1(
            pred_logits=logits_cat,
            label_ids=labels_cat,
            offset_mapping_batch=all_offsets,
        )

    return f1


# ======================================================================
#                       OPTUNA OBJECTIVE
# ======================================================================
def make_objective(
    base_encoder: nn.Module,
    train_ds,
    val_ds,
    seed: int,
):
    """Return a closure for ``study.optimize()``."""

    def objective(trial: optuna.Trial) -> float:
        model = None
        optimizer = None
        scheduler = None
        scaler = None

        try:
            # ── 1. Sample hyperparameters ────────────────────────────
            head_choices = ["linear", "mlp"]
            if CRF_AVAILABLE:
                head_choices.append("crf")
            head_type = trial.suggest_categorical("head_type", head_choices)

            lr_encoder = trial.suggest_float("lr_encoder", 1e-6, 5e-5, log=True)
            lr_head = trial.suggest_float("lr_head", 1e-5, 1e-3, log=True)
            weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
            dropout = trial.suggest_float("dropout", 0.0, 0.5)
            batch_size = trial.suggest_categorical("batch_size", [16, 24, 32])
            sched_type = trial.suggest_categorical("scheduler", ["linear", "cosine", "none"])
            warmup_frac = trial.suggest_float("warmup_frac", 0.0, 0.2)
            max_grad_norm = trial.suggest_float("max_grad_norm", 0.0, 1.0)

            # MLP‑specific
            mlp_layers = 1
            mlp_hidden = 512
            activation = "gelu"
            if head_type == "mlp":
                mlp_layers = trial.suggest_int("mlp_layers", 1, 3)
                mlp_hidden = trial.suggest_int("mlp_hidden", 256, 1024, step=64)
                activation = trial.suggest_categorical("activation", ["gelu", "relu"])

            # ── 2. Gradient accumulation ─────────────────────────────
            per_device_batch = min(batch_size, MAX_GPU_BATCH)
            accum_steps = math.ceil(batch_size / per_device_batch)

            # ── 3. Build model ───────────────────────────────────────
            model_seed = seed + trial.number
            torch.manual_seed(model_seed)
            random.seed(model_seed)
            np.random.seed(model_seed)

            encoder = copy.deepcopy(base_encoder)
            model = NERModel(
                encoder=encoder,
                head_type=head_type,
                num_labels=NUM_LABELS,
                dropout=dropout,
                mlp_layers=mlp_layers,
                mlp_hidden=mlp_hidden,
                activation=activation,
            ).to(DEVICE)

            # ── 4. DataLoaders (deterministic shuffle) ───────────────
            crf_mode = head_type == "crf"
            collate = lambda batch: ner_collate_fn(batch, crf_mode=crf_mode)

            g_train = torch.Generator()
            g_train.manual_seed(seed)

            train_loader = DataLoader(
                train_ds,
                batch_size=per_device_batch,
                shuffle=True,
                collate_fn=collate,
                num_workers=2,
                pin_memory=True,
                generator=g_train,
                drop_last=False,
            )
            val_loader = DataLoader(
                val_ds,
                batch_size=per_device_batch * 2,
                shuffle=False,
                collate_fn=collate,
                num_workers=2,
                pin_memory=True,
            )

            # ── 5. Optimizer (separate LR, exclude bias/LN from WD) ─
            no_decay_keys = ["bias", "LayerNorm.weight", "layer_norm.weight"]

            encoder_decay, encoder_no_decay = [], []
            head_decay, head_no_decay = [], []
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue
                is_encoder = name.startswith("encoder.")
                is_no_decay = any(k in name for k in no_decay_keys)
                if is_encoder:
                    (encoder_no_decay if is_no_decay else encoder_decay).append(param)
                else:
                    (head_no_decay if is_no_decay else head_decay).append(param)

            optimizer = AdamW(
                [
                    {"params": encoder_decay, "lr": lr_encoder, "weight_decay": weight_decay},
                    {"params": encoder_no_decay, "lr": lr_encoder, "weight_decay": 0.0},
                    {"params": head_decay, "lr": lr_head, "weight_decay": weight_decay},
                    {"params": head_no_decay, "lr": lr_head, "weight_decay": 0.0},
                ]
            )

            # ── 6. Scheduler ────────────────────────────────────────
            n_batches = len(train_loader)
            optimizer_steps_per_epoch = math.ceil(n_batches / accum_steps)
            total_optimizer_steps = optimizer_steps_per_epoch * NUM_EPOCHS
            warmup_steps = int(total_optimizer_steps * warmup_frac)

            scheduler = None  # type: ignore[assignment]
            if sched_type == "linear":
                scheduler = get_linear_schedule_with_warmup(
                    optimizer, warmup_steps, total_optimizer_steps
                )
            elif sched_type == "cosine":
                scheduler = get_cosine_schedule_with_warmup(
                    optimizer, warmup_steps, total_optimizer_steps
                )
            # sched_type == "none" → scheduler stays None

            scaler = GradScaler()

            # ── 7. Training loop ────────────────────────────────────
            best_f1 = 0.0

            for epoch in range(NUM_EPOCHS):
                model.train()
                epoch_loss = 0.0
                n_loss_steps = 0
                optimizer.zero_grad()

                for step, batch in enumerate(train_loader, start=1):
                    input_ids = batch["input_ids"].to(DEVICE, non_blocking=True)
                    attn_mask = batch["attention_mask"].to(DEVICE, non_blocking=True)
                    labels_t = batch["labels"].to(DEVICE, non_blocking=True)

                    with autocast():
                        out = model(input_ids, attn_mask, labels=labels_t)
                        loss = out["loss"] / accum_steps  # scale for accumulation

                    scaler.scale(loss).backward()

                    if step % accum_steps == 0:
                        if max_grad_norm > 0:
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(
                                model.parameters(), max_grad_norm
                            )
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                        if scheduler is not None:
                            scheduler.step()

                    epoch_loss += loss.item() * accum_steps  # unscaled
                    n_loss_steps += 1

                # ── Flush remaining accumulated gradients ────────────
                if step % accum_steps != 0:
                    if max_grad_norm > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), max_grad_norm
                        )
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    if scheduler is not None:
                        scheduler.step()

                # ── Evaluate ─────────────────────────────────────────
                avg_loss = epoch_loss / max(n_loss_steps, 1)
                f1 = evaluate(model, val_loader, DEVICE)

                log.info(
                    "Trial %d | Epoch %d/%d | loss %.4f | val_f1 %.4f",
                    trial.number,
                    epoch + 1,
                    NUM_EPOCHS,
                    avg_loss,
                    f1,
                )

                trial.report(f1, epoch)
                if trial.should_prune():
                    log.info("Trial %d pruned at epoch %d.", trial.number, epoch + 1)
                    raise optuna.TrialPruned()

                if f1 > best_f1:
                    best_f1 = f1

            log.info(
                "Trial %d finished — best val_f1 %.4f", trial.number, best_f1
            )
            return best_f1

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                log.warning(
                    "Trial %d OOM — pruning. Error: %s", trial.number, e
                )
                torch.cuda.empty_cache()
                gc.collect()
                raise optuna.TrialPruned()
            raise

        finally:
            # Systematic VRAM cleanup between trials
            del model, optimizer, scheduler, scaler
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return objective


# ======================================================================
#                             MAIN
# ======================================================================
def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Optuna hyperparameter search for CamemBERT NER."
    )
    p.add_argument(
        "--dataset",
        type=str,
        default="./prepared_data",
        help="Path to HF DatasetDict produced by prepare_data.py.",
    )
    p.add_argument("--n-trials", type=int, default=30, help="Number of Optuna trials.")
    p.add_argument("--seed", type=int, default=42, help="Global random seed.")
    p.add_argument(
        "--trials-dir",
        type=str,
        default="./optuna_results",
        help="Directory for Optuna artefacts (best_trial.json, trials history).",
    )
    p.add_argument(
        "--subsample-frac",
        type=float,
        default=0.2,
        help="Fraction of the *training set* to use for search (default 0.2).",
    )
    return p.parse_args(argv)


def main(argv: List[str] | None = None) -> None:
    args = parse_args(argv)
    log.info("Arguments: %s", vars(args))

    # ── Reproducibility ─────────────────────────────────────────────
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    log.info("Device: %s", DEVICE)
    if torch.cuda.is_available():
        log.info(
            "GPU: %s | VRAM: %.1f GB",
            torch.cuda.get_device_name(0),
            torch.cuda.get_device_properties(0).total_memory / 1e9,
        )

    # ── Load dataset ────────────────────────────────────────────────
    log.info("Loading dataset from: %s", args.dataset)
    ds = DatasetDict.load_from_disk(args.dataset)
    train_ds = ds["train"]
    val_ds = ds["val"]
    log.info("Full train size: %d | val size: %d", len(train_ds), len(val_ds))

    # ── Subsample training set ──────────────────────────────────────
    if 0.0 < args.subsample_frac < 1.0:
        n_sub = max(1, int(len(train_ds) * args.subsample_frac))
        rng = np.random.RandomState(args.seed)
        sub_indices = rng.choice(len(train_ds), size=n_sub, replace=False).tolist()
        train_ds = train_ds.select(sub_indices)
        log.info(
            "Subsampled train to %d rows (frac=%.2f)", len(train_ds), args.subsample_frac
        )

    # ── Load base encoder (once) ────────────────────────────────────
    log.info("Loading base encoder: %s", MODEL_NAME)
    base_encoder = AutoModel.from_pretrained(MODEL_NAME)
    base_encoder.eval()  # will be deepcopied per trial, then .train()
    log.info(
        "Encoder loaded — hidden_size=%d, layers=%d",
        base_encoder.config.hidden_size,
        base_encoder.config.num_hidden_layers,
    )

    # ── Head choices ────────────────────────────────────────────────
    head_choices = ["linear", "mlp"]
    if CRF_AVAILABLE:
        head_choices.append("crf")
        log.info("CRF available — search space includes 'crf' head.")
    else:
        log.info("CRF not available — search space limited to 'linear' and 'mlp'.")

    # ── Optuna study ────────────────────────────────────────────────
    sampler = optuna.samplers.TPESampler(seed=args.seed)
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=2)

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        study_name="camembert_ner_hpo",
    )

    objective_fn = make_objective(
        base_encoder=base_encoder,
        train_ds=train_ds,
        val_ds=val_ds,
        seed=args.seed,
    )

    log.info("Starting Optuna search — %d trials …", args.n_trials)
    t0 = time.time()
    study.optimize(objective_fn, n_trials=args.n_trials)
    elapsed = time.time() - t0
    log.info("Search completed in %.1f s (%.1f min).", elapsed, elapsed / 60)

    # ── Save results ────────────────────────────────────────────────
    out_dir = Path(args.trials_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Best trial
    if len(study.trials) == 0 or all(
        t.state != optuna.trial.TrialState.COMPLETE for t in study.trials
    ):
        log.error("No completed trials — cannot extract best trial.")
        sys.exit(1)

    best = study.best_trial
    best_info = {
        "trial_number": best.number,
        "value": best.value,
        "params": best.params,
        "duration_seconds": (best.datetime_complete - best.datetime_start).total_seconds()
        if best.datetime_complete and best.datetime_start
        else None,
    }
    best_path = out_dir / "best_trial.json"
    with open(best_path, "w") as f:
        json.dump(best_info, f, indent=2)
    log.info("Best trial saved to %s", best_path)
    log.info(
        "  → Trial %d | val_f1 = %.4f | params = %s",
        best.number,
        best.value,
        json.dumps(best.params, indent=2),
    )

    # All trials
    trials_history = []
    for t in study.trials:
        trials_history.append(
            {
                "number": t.number,
                "state": t.state.name,
                "value": t.value,
                "params": t.params,
                "duration_seconds": (
                    (t.datetime_complete - t.datetime_start).total_seconds()
                    if t.datetime_complete and t.datetime_start
                    else None
                ),
            }
        )
    history_path = out_dir / "all_trials.json"
    with open(history_path, "w") as f:
        json.dump(trials_history, f, indent=2)
    log.info("Full trial history saved to %s (%d trials)", history_path, len(trials_history))

    # Summary
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    log.info(
        "Summary: %d completed, %d pruned, %d failed.",
        len(completed),
        len(pruned),
        len(study.trials) - len(completed) - len(pruned),
    )


if __name__ == "__main__":
    main()
