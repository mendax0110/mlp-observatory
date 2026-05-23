from __future__ import annotations

import asyncio
import random
import time
from dataclasses import asdict
from typing import Any, Awaitable, Callable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, TensorDataset

from mlp_observatory.domain.events import RunEvent
from mlp_observatory.domain.models import EpochMetrics, ProjectConfig, TaskType
from mlp_observatory.training.model import ModelStrategy

PublishFn = Callable[[RunEvent], Awaitable[None]]


class Trainer:
    def __init__(self, model: ModelStrategy, config: ProjectConfig, device: torch.device) -> None:
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.task = config.task

        if self.task == TaskType.regression:
            self.criterion: nn.Module = nn.MSELoss()
        else:
            self.criterion = nn.BCEWithLogitsLoss()

        self.optimizer = self._build_optimizer()
        self.scheduler: torch.optim.lr_scheduler.LRScheduler | None = None
        self._scheduler_kind = config.train.scheduler

        self._amp_enabled = bool(config.train.mixed_precision and device.type == "cuda")
        self._scaler = torch.amp.GradScaler(self.device, enabled=self._amp_enabled)

    def _build_optimizer(self) -> torch.optim.Optimizer:
        cfg = self.config.train
        params = self.model.parameters()
        if cfg.optimizer == "sgd":
            return torch.optim.SGD(
                params,
                lr=cfg.learning_rate,
                momentum=cfg.momentum,
                weight_decay=(0.0 if cfg.weight_decay_decoupled else cfg.weight_decay),
            )
        if cfg.optimizer == "rmsprop":
            return torch.optim.RMSprop(
                params,
                lr=cfg.learning_rate,
                momentum=cfg.momentum,
                weight_decay=(0.0 if cfg.weight_decay_decoupled else cfg.weight_decay),
            )
        return torch.optim.AdamW(params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    def _build_scheduler(self, steps_per_epoch: int) -> None:
        cfg = self.config.train
        if cfg.scheduler == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=cfg.epochs)
        elif cfg.scheduler == "step":
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=cfg.scheduler_step_size,
                gamma=cfg.scheduler_gamma,
            )
        elif cfg.scheduler == "one_cycle":
            max_lr = cfg.one_cycle_max_lr or (cfg.learning_rate * 10.0)
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=max_lr,
                epochs=cfg.epochs,
                steps_per_epoch=max(1, steps_per_epoch),
            )
        else:
            self.scheduler = None

    def _split(self, dataset: TensorDataset) -> tuple[Subset[tuple[torch.Tensor, torch.Tensor]], Subset[tuple[torch.Tensor, torch.Tensor]]]:
        train_size = int(len(dataset) * self.config.data.train_ratio)
        val_size = len(dataset) - train_size
        return torch.utils.data.random_split(
            dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(self.config.train.seed),
        )

    def _normalize_input(self, dataset: TensorDataset) -> TensorDataset:
        if not self.config.train.input_normalization:
            return dataset
        x, y = dataset.tensors
        mean = x.mean(dim=0, keepdim=True)
        std = x.std(dim=0, keepdim=True)
        std = torch.where(std < 1e-6, torch.ones_like(std), std)
        x_norm = (x - mean) / std
        return TensorDataset(x_norm, y)

    async def fit(
        self,
        run_id: str,
        dataset: TensorDataset,
        publish: PublishFn,
        update_every_steps: int,
    ) -> tuple[list[EpochMetrics], dict[str, Any], dict[str, Any]]:
        self._set_seed(self.config.train.seed)
        dataset = self._normalize_input(dataset)
        train_set, val_set = self._split(dataset)
        train_loader = DataLoader(train_set, batch_size=self.config.train.batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=self.config.train.batch_size, shuffle=False)
        self._build_scheduler(len(train_loader))

        history: list[EpochMetrics] = []
        diag_acc = {
            "dead_ratio_sum": [],
            "grad_norm_sum": [],
            "weight_update_sum": [],
            "steps": 0,
        }

        best_val = float("inf")
        bad_epochs = 0

        for epoch in range(self.config.train.epochs):
            train_stats = await self._train_epoch(run_id, epoch, train_loader, publish, update_every_steps, diag_acc)
            val_stats = self._validate_epoch(val_loader)

            metrics = EpochMetrics(
                epoch=epoch + 1,
                train_loss=train_stats["train_loss"],
                val_loss=val_stats["val_loss"],
                val_accuracy=val_stats["val_score"],
                extras={
                    "grad_norm_mean": train_stats["grad_norm_mean"],
                    "weight_norm_mean": train_stats["weight_norm_mean"],
                    "update_ratio_mean": train_stats["update_ratio_mean"],
                    "dead_ratio_mean": train_stats["dead_ratio_mean"],
                    "saturation": train_stats["saturation"],
                    "sharpness_proxy": train_stats["sharpness_proxy"],
                    "weight_histograms": train_stats["weight_histograms"],
                    "grad_histograms": train_stats["grad_histograms"],
                    "train_activation_histograms": train_stats["train_activation_histograms"],
                    "activation_snapshot": train_stats["activation_snapshot"],
                    "val_activation_histograms": val_stats["val_activation_histograms"],
                    "confidence_calibration": val_stats["confidence_calibration"],
                    "lr": self.optimizer.param_groups[0].get("lr", self.config.train.learning_rate),
                },
            )
            history.append(metrics)
            await publish(RunEvent("epoch_end", run_id, asdict(metrics)))

            if self.scheduler is not None and self._scheduler_kind in {"cosine", "step"}:
                self.scheduler.step()

            patience = self.config.train.early_stopping_patience
            if patience is not None:
                min_delta = float(self.config.train.early_stopping_min_delta)
                if metrics.val_loss < (best_val - min_delta):
                    best_val = metrics.val_loss
                    bad_epochs = 0
                else:
                    bad_epochs += 1
                if bad_epochs >= patience:
                    await publish(RunEvent("early_stopped", run_id, {"epoch": epoch + 1, "patience": patience}))
                    break

        diagnostics_summary = self._finalize_diagnostics(diag_acc)
        final = {"feature_importance": self._feature_importance()}
        return history, diagnostics_summary, final

    async def _train_epoch(
        self,
        run_id: str,
        epoch: int,
        train_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
        publish: PublishFn,
        update_every_steps: int,
        diag_acc: dict[str, Any],
    ) -> dict[str, Any]:
        self.model.train()
        total = 0.0
        batches = 0
        last_update = 0.0
        min_update_interval_s = 0.2

        grad_norm_acc: list[list[float]] = []
        weight_norm_acc: list[list[float]] = []
        update_ratio_acc: list[list[float]] = []
        dead_ratio_acc: list[list[float]] = []
        sharpness_acc: list[float] = []
        sat_near_zero_acc: list[list[float]] = []
        sat_high_mag_acc: list[list[float]] = []
        latest_weight_hist: list[list[float]] = []
        latest_grad_hist: list[list[float]] = []
        latest_activation_hist: list[list[float]] = []
        latest_activation_means: list[list[float]] = []
        latest_activation_stds: list[list[float]] = []
        latest_weight_sample: list[float] = []
        latest_forward_trace: list[dict[str, object]] = []

        for step, (x, y) in enumerate(train_loader, start=1):
            x = x.to(self.device)
            y = y.to(self.device)

            prev_weights = [p.detach().clone() for p in self.model.parameters() if p.requires_grad]

            with torch.cuda.amp.autocast(enabled=self._amp_enabled):
                logits, hidden_acts, forward_trace = self.model.forward_with_diagnostics(
                    x,
                    trace_sample_index=self.config.train.trace_sample_index,
                )
                target = self._smooth_targets(y)
                loss = self.criterion(logits, target)
                loss = loss + self._regularization_penalty()

            self.optimizer.zero_grad(set_to_none=True)
            if self._amp_enabled:
                self._scaler.scale(loss).backward()
                if self.config.train.gradient_clip_norm:
                    self._scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.train.gradient_clip_norm)
                self._scaler.step(self.optimizer)
                self._scaler.update()
            else:
                loss.backward()
                if self.config.train.gradient_clip_norm:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.train.gradient_clip_norm)
                self.optimizer.step()

            if self.scheduler is not None and self._scheduler_kind == "one_cycle":
                self.scheduler.step()

            grad_norms = self._layer_grad_norms()
            dead_ratios = self._dead_neuron_ratios(hidden_acts)
            saturation = self._saturation_stats(hidden_acts)
            activation_means = [a.detach().mean(dim=0)[:24].cpu().tolist() for a in hidden_acts]
            activation_stds = [a.detach().std(dim=0)[:24].cpu().tolist() for a in hidden_acts]
            activation_hists = [self._histogram(a) for a in hidden_acts]

            latest_activation_means = activation_means
            latest_activation_stds = activation_stds
            latest_activation_hist = activation_hists
            latest_forward_trace = forward_trace
            latest_weight_sample = self._sample_weights()

            weight_norms = self._layer_weight_norms()
            weight_updates = self._weight_update_norms(prev_weights)
            update_ratios = [u / (w + 1e-8) for u, w in zip(weight_updates, weight_norms)]

            latest_weight_hist = self._param_histograms(use_grad=False)
            latest_grad_hist = self._param_histograms(use_grad=True)

            total += float(loss.item())
            batches += 1

            diag_acc = self._accumulate(diag_acc, dead_ratios, grad_norms, weight_updates)
            grad_norm_acc.append(grad_norms)
            weight_norm_acc.append(weight_norms)
            update_ratio_acc.append(update_ratios)
            dead_ratio_acc.append(dead_ratios)
            sharpness_acc.append(float(sum(grad_norms) / max(1, len(grad_norms))))
            sat_near_zero_acc.append(saturation["near_zero"])
            sat_high_mag_acc.append(saturation["high_mag"])

            if step % update_every_steps == 0:
                now = time.monotonic()
                if now - last_update >= min_update_interval_s:
                    last_update = now
                    payload = {
                        "epoch": epoch + 1,
                        "step": step,
                        "train_loss": float(loss.item()),
                        "weights": latest_weight_sample,
                        "activations": activation_means,
                        "activation_stds": activation_stds,
                        "activation_histograms": activation_hists,
                        "grad_norms": grad_norms,
                        "dead_neuron_ratio": dead_ratios,
                        "weight_update_norms": weight_updates,
                        "weight_norms": weight_norms,
                        "update_ratios": update_ratios,
                        "saturation": saturation,
                        "forward_trace": forward_trace,
                    }
                    await publish(RunEvent("batch_update", run_id, payload))

            await asyncio.sleep(0)

        return {
            "train_loss": total / max(1, batches),
            "grad_norm_mean": self._mean_vector(grad_norm_acc),
            "weight_norm_mean": self._mean_vector(weight_norm_acc),
            "update_ratio_mean": self._mean_vector(update_ratio_acc),
            "dead_ratio_mean": self._mean_vector(dead_ratio_acc),
            "saturation": {
                "near_zero": self._mean_vector(sat_near_zero_acc),
                "high_mag": self._mean_vector(sat_high_mag_acc),
            },
            "sharpness_proxy": float(sum(sharpness_acc) / max(1, len(sharpness_acc))),
            "weight_histograms": latest_weight_hist,
            "grad_histograms": latest_grad_hist,
            "train_activation_histograms": latest_activation_hist,
            "activation_snapshot": {
                "weights": latest_weight_sample,
                "activations": latest_activation_means,
                "activation_stds": latest_activation_stds,
                "forward_trace": latest_forward_trace,
            },
        }

    def _validate_epoch(self, val_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]]) -> dict[str, Any]:
        self.model.eval()
        total_loss = 0.0
        batches = 0
        val_acts_hists: list[list[float]] | None = None

        with torch.no_grad():
            if self.task == TaskType.regression:
                preds: list[torch.Tensor] = []
                targets: list[torch.Tensor] = []
                for x, y in val_loader:
                    x = x.to(self.device)
                    y = y.to(self.device)
                    logits, hidden, _ = self.model.forward_with_diagnostics(x, self.config.train.trace_sample_index)
                    if val_acts_hists is None:
                        val_acts_hists = [self._histogram(a) for a in hidden]
                    loss = self.criterion(logits, y)

                    preds.append(logits.detach().cpu())
                    targets.append(y.detach().cpu())
                    total_loss += float(loss.item())
                    batches += 1

                pred = torch.cat(preds, dim=0) if preds else torch.zeros((0, 1))
                target = torch.cat(targets, dim=0) if targets else torch.zeros((0, 1))
                score = self._r2_score(target, pred)
                return {
                    "val_loss": total_loss / max(1, batches),
                    "val_score": score,
                    "val_activation_histograms": val_acts_hists or [],
                    "confidence_calibration": [],
                }

            correct = 0
            samples = 0
            confidence_bins = np.zeros(10, dtype=np.float64)
            confidence_counts = np.zeros(10, dtype=np.float64)

            for x, y in val_loader:
                x = x.to(self.device)
                y = y.to(self.device)
                logits, hidden, _ = self.model.forward_with_diagnostics(x, self.config.train.trace_sample_index)
                if val_acts_hists is None:
                    val_acts_hists = [self._histogram(a) for a in hidden]
                loss = self.criterion(logits, y)

                probs = torch.sigmoid(logits)
                pred = (probs >= 0.5).float()
                correct += int((pred == y).sum().item())
                samples += int(y.numel())

                conf = torch.maximum(probs, 1.0 - probs).detach().cpu().flatten().numpy()
                acc = (pred == y).detach().cpu().flatten().numpy().astype(np.float64)
                idxs = np.clip((conf * 10).astype(int), 0, 9)
                for i, a in zip(idxs, acc):
                    confidence_bins[i] += a
                    confidence_counts[i] += 1.0

                total_loss += float(loss.item())
                batches += 1

            cal = []
            for i in range(10):
                center = (i + 0.5) / 10.0
                if confidence_counts[i] > 0:
                    cal.append({"bin": center, "accuracy": float(confidence_bins[i] / confidence_counts[i]), "count": int(confidence_counts[i])})
                else:
                    cal.append({"bin": center, "accuracy": 0.0, "count": 0})

        return {
            "val_loss": total_loss / max(1, batches),
            "val_score": (correct / max(1, samples)),
            "val_activation_histograms": val_acts_hists or [],
            "confidence_calibration": cal,
        }

    def _regularization_penalty(self) -> torch.Tensor:
        l1_lambda = float(self.config.train.l1_lambda)
        l2_lambda = float(self.config.train.l2_lambda)
        if l1_lambda <= 0 and l2_lambda <= 0:
            return torch.zeros((), device=self.device)

        l1 = torch.zeros((), device=self.device)
        l2 = torch.zeros((), device=self.device)
        for p in self.model.parameters():
            if l1_lambda > 0:
                l1 = l1 + p.abs().sum()
            if l2_lambda > 0:
                l2 = l2 + (p.pow(2).sum())
        return l1_lambda * l1 + l2_lambda * l2

    def _smooth_targets(self, y: torch.Tensor) -> torch.Tensor:
        if self.task != TaskType.binary_classification:
            return y
        eps = float(self.config.train.label_smoothing)
        if eps <= 0:
            return y
        return y * (1.0 - eps) + 0.5 * eps

    def _sample_weights(self) -> list[float]:
        values: list[float] = []
        for param in self.model.parameters():
            flat = param.detach().flatten()
            take = min(64, flat.numel())
            if take > 0:
                values.extend(flat[:take].cpu().tolist())
            if len(values) >= 256:
                break
        return values[:256]

    @staticmethod
    def _histogram(x: torch.Tensor) -> list[float]:
        flat = x.detach().flatten().cpu()
        hist = torch.histc(flat, bins=12, min=-3.0, max=3.0)
        hist = hist / max(1.0, float(hist.sum()))
        return hist.tolist()

    @staticmethod
    def _dead_neuron_ratios(hidden_acts: list[torch.Tensor]) -> list[float]:
        ratios: list[float] = []
        for act in hidden_acts:
            mean_abs_per_neuron = act.detach().abs().mean(dim=0)
            dead = (mean_abs_per_neuron < 1e-2).float().mean().item()
            ratios.append(float(dead))
        return ratios

    @staticmethod
    def _saturation_stats(hidden_acts: list[torch.Tensor]) -> dict[str, list[float]]:
        near_zero: list[float] = []
        high_mag: list[float] = []
        for act in hidden_acts:
            a = act.detach().abs()
            near_zero.append(float((a < 0.05).float().mean().item()))
            high_mag.append(float((a > 2.0).float().mean().item()))
        return {"near_zero": near_zero, "high_mag": high_mag}

    def _layer_grad_norms(self) -> list[float]:
        norms: list[float] = []
        for p in self.model.parameters():
            if p.grad is None:
                continue
            norms.append(float(p.grad.detach().norm().item()))
        return norms[:24]

    def _layer_weight_norms(self) -> list[float]:
        norms: list[float] = []
        for p in self.model.parameters():
            norms.append(float(p.detach().norm().item()))
        return norms[:24]

    def _weight_update_norms(self, prev_weights: list[torch.Tensor]) -> list[float]:
        updates: list[float] = []
        cur_weights = [p.detach() for p in self.model.parameters() if p.requires_grad]
        for old, new in zip(prev_weights, cur_weights):
            updates.append(float((new - old).norm().item()))
        return updates[:24]

    def _param_histograms(self, use_grad: bool) -> list[list[float]]:
        hists: list[list[float]] = []
        for p in self.model.parameters():
            if p.ndim < 2:
                continue
            tensor = p.grad.detach() if use_grad and p.grad is not None else p.detach()
            hists.append(self._histogram(tensor))
            if len(hists) >= 8:
                break
        return hists

    @staticmethod
    def _mean_vector(rows: list[list[float]]) -> list[float]:
        if not rows:
            return []
        width = max(len(r) for r in rows)
        out = [0.0] * width
        counts = [0] * width
        for row in rows:
            for i, v in enumerate(row):
                out[i] += float(v)
                counts[i] += 1
        for i in range(width):
            if counts[i] > 0:
                out[i] /= counts[i]
        return out

    @staticmethod
    def _accumulate(
        acc: dict[str, Any],
        dead_ratios: list[float],
        grad_norms: list[float],
        weight_updates: list[float],
    ) -> dict[str, Any]:
        acc["steps"] += 1

        while len(acc["dead_ratio_sum"]) < len(dead_ratios):
            acc["dead_ratio_sum"].append(0.0)
        while len(acc["grad_norm_sum"]) < len(grad_norms):
            acc["grad_norm_sum"].append(0.0)
        while len(acc["weight_update_sum"]) < len(weight_updates):
            acc["weight_update_sum"].append(0.0)

        for i, v in enumerate(dead_ratios):
            acc["dead_ratio_sum"][i] += v
        for i, v in enumerate(grad_norms):
            acc["grad_norm_sum"][i] += v
        for i, v in enumerate(weight_updates):
            acc["weight_update_sum"][i] += v
        return acc

    @staticmethod
    def _finalize_diagnostics(acc: dict[str, Any]) -> dict[str, Any]:
        steps = max(1, int(acc["steps"]))
        return {
            "avg_dead_neuron_ratio": [v / steps for v in acc["dead_ratio_sum"]],
            "avg_grad_norm": [v / steps for v in acc["grad_norm_sum"]],
            "avg_weight_update_norm": [v / steps for v in acc["weight_update_sum"]],
            "steps": steps,
        }

    def _feature_importance(self) -> list[float]:
        # linear: output weight directly. MLP: first layer absolute mean by input feature.
        with torch.no_grad():
            params = [p for p in self.model.parameters() if p.ndim == 2]
            if not params:
                return []
            first = params[0].detach().abs().cpu()
            if first.shape[0] == 1:
                return first.flatten().tolist()[:64]
            return first.mean(dim=0).flatten().tolist()[:64]

    @staticmethod
    def _r2_score(target: torch.Tensor, pred: torch.Tensor) -> float:
        if target.numel() == 0:
            return 0.0
        y = target.flatten().float()
        y_hat = pred.flatten().float()
        ss_res = torch.sum((y - y_hat) ** 2)
        y_mean = torch.mean(y)
        ss_tot = torch.sum((y - y_mean) ** 2)
        if float(ss_tot.item()) <= 1e-12:
            return 0.0
        return float((1.0 - (ss_res / ss_tot)).item())

    @staticmethod
    def _set_seed(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
