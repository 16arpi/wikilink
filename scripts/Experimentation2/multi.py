#!/usr/bin/env python3
"""
optimize_hyperparams.py – Optuna HPO for CamemBERT-av2 NER, multi-GPU.

Parallélisation trial-level sur N GPUs via torch.multiprocessing.spawn
+ Optuna SQLite storage pour la coordination inter-processus.

Usage
-----
    python optimize_hyperparams.py \
        --dataset ./prepared_data \
        --n-trials 30 \
        --seed 42 \
        --trials-dir ./optuna_results \
        --subsample-frac 0.2 \
        --n-gpus 2
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
from functools import partial                         # ← FIX 1 : import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import optuna
import torch
import torch.multiprocessing as mp
import torch.nn as nn
from datasets import DatasetDict
from seqeval.metrics import f1_score as seqeval_f1
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    AutoModel,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

# ──────────────────────────────────────────────────────────────────────
# Optional CRF
# ──────────────────────────────────────────────────────────────────────
CRF_AVAILABLE = False
try:
    from torchcrf import CRF

    CRF_AVAILABLE = True
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
PAD_TOKEN_ID = 0
PAD_LABEL_ID = -100
NUM_EPOCHS = 5

# ← CHANGÉ : DeBERTaV2
MODEL_NAME = "almanach/camembertav2-base"

# ← CHANGÉ : réduit pour DeBERTaV2 (attention désenchevêtrée = +VRAM)
MAX_GPU_BATCH = 20

# ← NOUVEAU : tuning DataLoader pour saturer le pipeline CPU→GPU
DATALOADER_WORKERS = 2        # par DataLoader ; 2 workers × 2 loaders × 2 GPU = 8 proc
PREFETCH_FACTOR = 4           # batches pré-chargés par worker

# ← SUPPRIMÉ : plus de DEVICE global, chaque worker reçoit le sien

# ──────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────
def _setup_worker_logging(gpu_id: int) -> logging.Logger:
    """Configure un logger propre au worker (évite les doublons mp)."""
    logger = logging.getLogger(f"worker-{gpu_id}")
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(
            f"%(asctime)s | GPU {gpu_id} | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


# Silence loggers bruyants au niveau du module
optuna.logging.set_verbosity(optuna.logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)


# ======================================================================
#                           NER MODEL
# ======================================================================
class NERModel(nn.Module):
    """CamemBERT encoder + tête de classification configurable.

    head_type: "linear" | "mlp" | "crf"
    """

    def __init__(
        self,
        encoder: nn.Module,
        head_type: str,
        num_labels: int = NUM_LABELS,
        dropout: float = 0.1,
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
                raise RuntimeError("CRF head requested but torchcrf not installed.")
            self.emission = nn.Linear(hidden_size, num_labels)
            self.crf = CRF(num_labels, batch_first=True)
        else:
            raise ValueError(f"Unknown head_type: {head_type!r}")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        hidden = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state
        hidden = self.drop(hidden)
        out: Dict[str, Any] = {}

        if self.head_type == "crf":
            emissions = self.emission(hidden)
            mask = attention_mask.bool()
            out["logits"] = emissions
            if labels is not None:
                out["loss"] = -self.crf(emissions, labels, mask=mask, reduction="mean")
            else:
                out["decoded"] = self.crf.decode(emissions, mask=mask)
        else:
            logits = self.classifier(hidden)
            out["logits"] = logits
            if labels is not None:
                out["loss"] = nn.CrossEntropyLoss(ignore_index=PAD_LABEL_ID)(
                    logits.view(-1, self.num_labels), labels.view(-1)
                )
        return out


# ======================================================================
#                          DATA COLLATOR
# ======================================================================
def ner_collate_fn(
    batch: List[Dict[str, Any]],
    pad_token_id: int = PAD_TOKEN_ID,
    crf_mode: bool = False,
) -> Dict[str, Any]:
    input_ids = [torch.tensor(ex["input_ids"], dtype=torch.long) for ex in batch]
    attention = [torch.tensor(ex["attention_mask"], dtype=torch.long) for ex in batch]
    labels = [torch.tensor(ex["labels"], dtype=torch.long) for ex in batch]
    offsets = [ex["offset_mapping"] for ex in batch]

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
            if tok_start == 0 and tok_end == 0:
                continue
            is_first_subtoken = tok_start > prev_end
            prev_end = tok_end
            if not is_first_subtoken:
                continue
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
    amp_dtype: torch.dtype,                       # ← NOUVEAU
) -> float:
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

        # ← CHANGÉ : nouvelle API autocast + dtype paramétrable
        with torch.amp.autocast(device_type="cuda", dtype=amp_dtype):
            out = model(input_ids, attn_mask)

        if is_crf:
            assert all_crf_decoded is not None
            all_crf_decoded.extend(out["decoded"])
        else:
            all_pred_logits.append(out["logits"].cpu())
        all_label_ids.append(labels.cpu())
        all_offsets.extend(offsets)

    if is_crf:
        labels_cat = torch.cat(all_label_ids, dim=0)
        return compute_entity_macro_f1(
            None, labels_cat, all_offsets, crf_decoded=all_crf_decoded
        )
    else:
        logits_cat = torch.cat(all_pred_logits, dim=0)
        labels_cat = torch.cat(all_label_ids, dim=0)
        return compute_entity_macro_f1(logits_cat, labels_cat, all_offsets)


# ======================================================================
#                       OPTUNA OBJECTIVE
# ======================================================================
def make_objective(
    base_encoder: nn.Module,
    train_ds,
    val_ds,
    seed: int,
    device: torch.device,          # ← NOUVEAU : plus de global DEVICE
    gpu_id: int,                   # ← NOUVEAU : pour logging
    amp_dtype: torch.dtype,        # ← NOUVEAU : FP16 ou BF16
    log: logging.Logger,           # ← NOUVEAU : logger du worker
):
    """Closure pour study.optimize()."""

    def objective(trial: optuna.Trial) -> float:
        model = None
        optimizer = None
        scheduler = None
        scaler = None

        try:
            # ── 1. Hyperparamètres ───────────────────────────────
            head_choices = ["linear", "mlp"]
            if CRF_AVAILABLE:
                head_choices.append("crf")
            head_type = trial.suggest_categorical("head_type", head_choices)

            lr_encoder = trial.suggest_float("lr_encoder", 1e-6, 5e-5, log=True)
            lr_head = trial.suggest_float("lr_head", 1e-5, 1e-3, log=True)
            weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
            dropout = trial.suggest_float("dropout", 0.0, 0.5)
            batch_size = trial.suggest_categorical("batch_size", [16, 20, 24])
            sched_type = trial.suggest_categorical("scheduler", ["linear", "cosine", "none"])
            warmup_frac = trial.suggest_float("warmup_frac", 0.0, 0.2)
            max_grad_norm = trial.suggest_float("max_grad_norm", 0.0, 1.0)

            # ← NOUVEAU : option gradient checkpointing dans l'espace de recherche
            use_grad_ckpt = trial.suggest_categorical("gradient_checkpointing", [True, False])

            mlp_layers = 1
            mlp_hidden = 512
            activation = "gelu"
            if head_type == "mlp":
                mlp_layers = trial.suggest_int("mlp_layers", 1, 3)
                mlp_hidden = trial.suggest_int("mlp_hidden", 256, 1024, step=64)
                activation = trial.suggest_categorical("activation", ["gelu", "relu"])

            # ── 2. Gradient accumulation ─────────────────────────
            per_device_batch = min(batch_size, MAX_GPU_BATCH)
            accum_steps = math.ceil(batch_size / per_device_batch)

            # ── 3. Construction du modèle ────────────────────────
            model_seed = seed + trial.number
            torch.manual_seed(model_seed)
            random.seed(model_seed)
            np.random.seed(model_seed)

            encoder = copy.deepcopy(base_encoder)

            # ← NOUVEAU : gradient checkpointing pour économiser la VRAM
            if use_grad_ckpt:
                encoder.gradient_checkpointing_enable()
                log.info(
                    "Trial %d: gradient checkpointing ENABLED", trial.number
                )

            model = NERModel(
                encoder=encoder,
                head_type=head_type,
                num_labels=NUM_LABELS,
                dropout=dropout,
                mlp_layers=mlp_layers,
                mlp_hidden=mlp_hidden,
                activation=activation,
            ).to(device)                             # ← CHANGÉ : device paramétré

            # ── 4. DataLoaders ───────────────────────────────────
            crf_mode = head_type == "crf"

            # ← FIX 1 : functools.partial au lieu de lambda
            #   Les lambdas imbriquées ne sont pas sérialisables (pickle)
            #   par les workers DataLoader lancés via mp.spawn ("spawn"
            #   start method).  functools.partial EST sérialisable.
            collate = partial(ner_collate_fn, crf_mode=crf_mode)

            g_train = torch.Generator()
            g_train.manual_seed(seed)

            # ← CHANGÉ : tuning DataLoader pour maximiser l'usage CPU/RAM
            # ← FIX 2 : supprimé pin_memory_device (déprécié PyTorch ≥ 2.x)
            dl_kwargs: Dict[str, Any] = dict(
                collate_fn=collate,
                num_workers=DATALOADER_WORKERS,
                pin_memory=True,
                persistent_workers=True,                 # ← pas de re-fork
                prefetch_factor=PREFETCH_FACTOR,         # ← pipeline saturé
            )

            train_loader = DataLoader(
                train_ds,
                batch_size=per_device_batch,
                shuffle=True,
                generator=g_train,
                drop_last=False,
                **dl_kwargs,
            )
            val_loader = DataLoader(
                val_ds,
                batch_size=per_device_batch * 2,
                shuffle=False,
                **dl_kwargs,
            )

            # ── 5. Optimiseur ────────────────────────────────────
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

            optimizer = AdamW([
                {"params": encoder_decay,    "lr": lr_encoder, "weight_decay": weight_decay},
                {"params": encoder_no_decay, "lr": lr_encoder, "weight_decay": 0.0},
                {"params": head_decay,       "lr": lr_head,    "weight_decay": weight_decay},
                {"params": head_no_decay,    "lr": lr_head,    "weight_decay": 0.0},
            ])

            # ── 6. Scheduler ────────────────────────────────────
            n_batches = len(train_loader)
            optimizer_steps_per_epoch = math.ceil(n_batches / accum_steps)
            total_optimizer_steps = optimizer_steps_per_epoch * NUM_EPOCHS
            warmup_steps = int(total_optimizer_steps * warmup_frac)

            scheduler = None
            if sched_type == "linear":
                scheduler = get_linear_schedule_with_warmup(
                    optimizer, warmup_steps, total_optimizer_steps
                )
            elif sched_type == "cosine":
                scheduler = get_cosine_schedule_with_warmup(
                    optimizer, warmup_steps, total_optimizer_steps
                )

            # ← CHANGÉ : nouvelle API GradScaler
            scaler = torch.amp.GradScaler("cuda", enabled=True)

            # ── 7. Boucle d'entraînement ────────────────────────
            best_f1 = 0.0
            nan_count = 0                             # ← NOUVEAU

            for epoch in range(NUM_EPOCHS):
                model.train()
                epoch_loss = 0.0
                n_loss_steps = 0
                optimizer.zero_grad()

                for step, batch in enumerate(train_loader, start=1):
                    input_ids = batch["input_ids"].to(device, non_blocking=True)
                    attn_mask = batch["attention_mask"].to(device, non_blocking=True)
                    labels_t = batch["labels"].to(device, non_blocking=True)

                    # ← CHANGÉ : nouvelle API autocast + dtype paramétrable
                    with torch.amp.autocast(
                        device_type="cuda", dtype=amp_dtype
                    ):
                        out = model(input_ids, attn_mask, labels=labels_t)
                        loss = out["loss"] / accum_steps

                    # ── NOUVEAU : garde NaN/Inf ──────────────────
                    if torch.isnan(loss) or torch.isinf(loss):
                        nan_count += 1
                        log.warning(
                            "Trial %d | NaN/Inf loss (count=%d)",
                            trial.number, nan_count,
                        )
                        if nan_count > 10:
                            log.warning(
                                "Trial %d pruned: trop de NaN", trial.number
                            )
                            raise optuna.TrialPruned()
                        optimizer.zero_grad()
                        continue
                    # ─────────────────────────────────────────────

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

                    epoch_loss += loss.item() * accum_steps
                    n_loss_steps += 1

                # Flush gradients résiduels
                if n_loss_steps > 0 and step % accum_steps != 0:
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

                avg_loss = epoch_loss / max(n_loss_steps, 1)
                f1 = evaluate(model, val_loader, device, amp_dtype)

                log.info(
                    "Trial %d | Epoch %d/%d | loss %.4f | val_f1 %.4f",
                    trial.number, epoch + 1, NUM_EPOCHS, avg_loss, f1,
                )

                trial.report(f1, epoch)
                if trial.should_prune():
                    log.info("Trial %d pruned at epoch %d.", trial.number, epoch + 1)
                    raise optuna.TrialPruned()

                if f1 > best_f1:
                    best_f1 = f1

            log.info("Trial %d finished — best val_f1 %.4f", trial.number, best_f1)
            return best_f1

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                log.warning("Trial %d OOM — pruning. %s", trial.number, e)
                torch.cuda.empty_cache()
                gc.collect()
                raise optuna.TrialPruned()
            raise

        finally:
            del model, optimizer, scheduler, scaler
            gc.collect()
            torch.cuda.empty_cache()

    return objective


# ======================================================================
#               NOUVEAU : WORKER GPU (1 par GPU)
# ======================================================================
def gpu_worker(
    rank: int,           # injecté par mp.spawn (0, 1, …)
    world_size: int,
    n_trials_per_gpu: int,
    args_dict: dict,     # Namespace → dict pour la sérialisation spawn
) -> None:
    """Chaque worker charge les données, l'encodeur, et exécute ses trials."""

    # ── 1. Pinning GPU ──────────────────────────────────────────────
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    log = _setup_worker_logging(rank)
    log.info("Worker %d démarré sur %s", rank, torch.cuda.get_device_name(rank))

    # ── 2. Détection du dtype AMP optimal ───────────────────────────
    #    T4 = Turing (sm_75)  → FP16 seulement
    #    A100/H100 = Ampere+  → BF16 (plus stable pour DeBERTaV2)
    #
    # ← FIX 3 : torch.cuda.is_bf16_supported() peut renvoyer True sur
    #   T4 dans les PyTorch récents (il vérifie le runtime CUDA, pas la
    #   compute capability du GPU).  On vérifie donc directement la
    #   compute capability : sm_80+ (Ampere) = BF16 natif.
    cap = torch.cuda.get_device_capability(rank)
    if cap[0] >= 8:                                   # Ampere sm_80+
        amp_dtype = torch.bfloat16
        log.info(
            "Compute capability %d.%d ≥ 8.0 — utilisation de bfloat16 pour AMP",
            cap[0], cap[1],
        )
    else:
        amp_dtype = torch.float16
        log.info(
            "Compute capability %d.%d < 8.0 — utilisation de float16 pour AMP",
            cap[0], cap[1],
        )

    # ── 3. Reproductibilité ─────────────────────────────────────────
    seed = args_dict["seed"]
    torch.manual_seed(seed + rank)
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.cuda.manual_seed(seed + rank)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # ── 4. Chargement dataset (memory-mapped Arrow → quasi gratuit) ─
    log.info("Chargement dataset: %s", args_dict["dataset"])
    ds = DatasetDict.load_from_disk(args_dict["dataset"])
    train_ds = ds["train"]
    val_ds = ds["val"]
    log.info("train=%d | val=%d", len(train_ds), len(val_ds))

    # Sous-échantillonnage (même indices pour tous les workers → comparable)
    subsample_frac = args_dict["subsample_frac"]
    if 0.0 < subsample_frac < 1.0:
        n_sub = max(1, int(len(train_ds) * subsample_frac))
        rng = np.random.RandomState(seed)  # même seed → mêmes indices
        sub_indices = rng.choice(len(train_ds), size=n_sub, replace=False).tolist()
        train_ds = train_ds.select(sub_indices)
        log.info("Sous-échantillon train: %d lignes (%.0f%%)",
                 len(train_ds), subsample_frac * 100)

    # ── 5. Chargement encodeur (une seule fois par worker) ──────────
    log.info("Chargement encodeur: %s", MODEL_NAME)
    base_encoder = AutoModel.from_pretrained(MODEL_NAME)
    base_encoder.eval()
    log.info(
        "Encodeur chargé — hidden=%d, layers=%d, type=%s",
        base_encoder.config.hidden_size,
        base_encoder.config.num_hidden_layers,
        base_encoder.config.model_type,
    )

    # ── 6. Connexion à l'étude Optuna partagée ──────────────────────
    storage_url = args_dict["storage_url"]
    study = optuna.load_study(
        study_name="camembert_ner_hpo",
        storage=storage_url,
        sampler=optuna.samplers.TPESampler(seed=seed + rank),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=2),
    )

    # ── 7. Lancement des trials ─────────────────────────────────────
    objective_fn = make_objective(
        base_encoder=base_encoder,
        train_ds=train_ds,
        val_ds=val_ds,
        seed=seed,
        device=device,
        gpu_id=rank,
        amp_dtype=amp_dtype,
        log=log,
    )

    log.info("Lancement de %d trials sur GPU %d", n_trials_per_gpu, rank)
    study.optimize(objective_fn, n_trials=n_trials_per_gpu)
    log.info("Worker %d terminé.", rank)


# ======================================================================
#                             MAIN
# ======================================================================
def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Optuna HPO multi-GPU pour CamemBERT-av2 NER."
    )
    p.add_argument("--dataset", type=str, default="./prepared_data")
    p.add_argument("--n-trials", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--trials-dir", type=str, default="./optuna_results")
    p.add_argument("--subsample-frac", type=float, default=0.2)
    # ← NOUVEAU
    p.add_argument(
        "--n-gpus", type=int, default=None,
        help="Nombre de GPUs (défaut: tous les GPUs visibles).",
    )
    return p.parse_args(argv)


def main(argv: List[str] | None = None) -> None:
    args = parse_args(argv)

    # ── Nombre de GPUs ──────────────────────────────────────────────
    n_gpus_available = torch.cuda.device_count()
    if n_gpus_available == 0:
        print("ERREUR: aucun GPU CUDA détecté.", file=sys.stderr)
        sys.exit(1)

    n_gpus = args.n_gpus if args.n_gpus else n_gpus_available
    n_gpus = min(n_gpus, n_gpus_available)

    # ── Répertoire de sortie ────────────────────────────────────────
    out_dir = Path(args.trials_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Création de l'étude Optuna (SQLite, partagée entre workers) ─
    storage_url = f"sqlite:///{out_dir.resolve()}/optuna_study.db"
    study = optuna.create_study(
        study_name="camembert_ner_hpo",
        storage=storage_url,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=args.seed),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=2),
        load_if_exists=True,
    )

    # ── Répartition des trials ──────────────────────────────────────
    n_trials_per_gpu = math.ceil(args.n_trials / n_gpus)

    print(f"╔══════════════════════════════════════════════════╗")
    print(f"║  Optuna HPO — {MODEL_NAME}")
    print(f"║  GPUs: {n_gpus} × {torch.cuda.get_device_name(0)}")
    print(f"║  Trials: {args.n_trials} ({n_trials_per_gpu}/GPU)")
    print(f"║  Storage: {storage_url}")
    print(f"╚══════════════════════════════════════════════════╝")

    # ── Dictionnaire sérialisable pour mp.spawn ─────────────────────
    args_dict = {
        "dataset": args.dataset,
        "seed": args.seed,
        "subsample_frac": args.subsample_frac,
        "storage_url": storage_url,
    }

    # ── Lancement parallèle ─────────────────────────────────────────
    t0 = time.time()

    if n_gpus == 1:
        # Pas besoin de spawn pour 1 GPU
        gpu_worker(0, 1, n_trials_per_gpu, args_dict)
    else:
        mp.spawn(
            gpu_worker,
            args=(n_gpus, n_trials_per_gpu, args_dict),
            nprocs=n_gpus,
            join=True,           # bloque jusqu'à ce que tous finissent
        )

    elapsed = time.time() - t0

    # ── Collecte des résultats (processus principal) ────────────────
    study = optuna.load_study(
        study_name="camembert_ner_hpo",
        storage=storage_url,
    )

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    failed = len(study.trials) - len(completed) - len(pruned)

    print(f"\n{'='*60}")
    print(f"Recherche terminée en {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"  Completed: {len(completed)} | Pruned: {len(pruned)} | Failed: {failed}")

    if not completed:
        print("ERREUR: aucun trial terminé.", file=sys.stderr)
        sys.exit(1)

    # Best trial
    best = study.best_trial
    best_info = {
        "trial_number": best.number,
        "value": best.value,
        "params": best.params,
        "duration_seconds": (
            (best.datetime_complete - best.datetime_start).total_seconds()
            if best.datetime_complete and best.datetime_start else None
        ),
    }
    best_path = out_dir / "best_trial.json"
    with open(best_path, "w") as f:
        json.dump(best_info, f, indent=2)
    print(f"\n  Best trial: #{best.number} | val_f1 = {best.value:.4f}")
    print(f"  Params: {json.dumps(best.params, indent=4)}")
    print(f"  → {best_path}")

    # All trials
    trials_history = []
    for t in study.trials:
        trials_history.append({
            "number": t.number,
            "state": t.state.name,
            "value": t.value,
            "params": t.params,
            "duration_seconds": (
                (t.datetime_complete - t.datetime_start).total_seconds()
                if t.datetime_complete and t.datetime_start else None
            ),
        })
    history_path = out_dir / "all_trials.json"
    with open(history_path, "w") as f:
        json.dump(trials_history, f, indent=2)
    print(f"  Historique: {history_path} ({len(trials_history)} trials)")
    print(f"{'='*60}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrompu.", file=sys.stderr)
        sys.exit(130)
