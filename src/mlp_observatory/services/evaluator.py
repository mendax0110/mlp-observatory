from __future__ import annotations
from dataclasses import replace, dataclass
from mlp_observatory.domain.models import (
    DataConfig, EpochMetrics, LayerConfig, ModelConfig, 
    ProjectConfig, TaskType, TrainConfig,
)

@dataclass(slots=True)
class Suggestion:
    message: str
    config_patch: ProjectConfig | None = None

class RunEvaluator:
    
    @staticmethod
    def evaluate(config: ProjectConfig, history: list[EpochMetrics], diagnostics: dict[str, object]) -> list[str]: 
        recs: list[str] = []
        recs.extend(RunEvaluator._check_config(config))
        
        if not history:
            recs.append("No history collected. Check training pipeline.")
            return recs
        recs.extend(RunEvaluator._check_training(config, history, diagnostics))
        if not recs:
            recs.append("Training looks healthy. Try increasing epochs for potentially better validation score.")
        return recs
    
    @staticmethod
    def suggest_config(config: ProjectConfig, history: list[EpochMetrics], diagnostics: dict[str, object]) -> list[Suggestion]:
        suggestions: list[Suggestion] = []
        suggestions.extend(RunEvaluator._suggest_from_config(config))
        if history:
            suggestions.extend(RunEvaluator._suggest_from_training(config, history, diagnostics))
        return suggestions
    
    @staticmethod
    def _estimate_params(config: ProjectConfig) -> int:
        if config.model.family != "mlp":
            return config.data.features + 1
        total = 0
        in_dim = config.data.features
        for layer in config.model.layers:
            total += in_dim * layer.units + layer.units
            in_dim = layer.units
        total += in_dim + 1
        return total
    
    @staticmethod
    def _check_config(config: ProjectConfig) -> list[str]:
        recs: list[str] = []
        t = config.train
        m = config.model
        d = config.data
        layers = m.layers if m.family == "mlp" else []
        
        if t.optimizer == "sgd" and t.learning_rate < 0.01:
            recs.append("SGD typically needs LR >= 0.01. Your LR may be too low for SGD to converge.")
        if t.optimizer == "adamw" and t.learning_rate > 0.01:
            recs.append("AdamW LR > 0.01 is unusally high and may cause instability.")
            
        if t.scheduler == "one_cycle" and t.one_cycle_max_lr is None:
            recs.append("OnceCycleLR selected but one_cycle_max_tr is not set, it will fall back to base LR")
        if t.scheduler == "step" and t.scheduler_step_size >= t.epochs:
            recs.append("Scheduler step_size >= epochs: the scheduler will never fire. Reduce step_size.")
            
        if d.samples > 0 and t.batch_size > d.samples * d.train_ratio:
            recs.append("Batch size exceeds training set size, effective batch = full dataset every step.")
        if t.batch_size < 32 and any(l.norm == "batchnorm" for l in layers):
            recs.append("BatchNorm with batch_size < 32 is unstable. Switch to LayerNorm.")
            
        if m.residual_every_2 and len(layers) < 4:
            recs.append("Residual connections have minimal effect with fewer than 4 layers.")

        for i, layer in enumerate(layers):
            if layer.dropout > 0.5 and layer.units < 64:
                recs.append(
                    f"Layer ({i+1}): dropout {layer.dropout} on {layer.units} unit is aggressive "
                    "and will discard most information."
                )
                
        total_params = RunEvaluator._estimate_params(config)
        if total_params > d.samples * 0.5:
            recs.append(
                f"Model has ~{total_params:,} params but only {d.samples} samples "
                " high risk of overfitting without strong regulariation."
            )
            
        if t.l1_lambda > 0 and t.l2_lambda > 0 and t.weight_decay > 0:
            recs.append("L1, L2 and weight_decay are all active. This is likely double-penalising weights.")
            
        if m.initialization == "xavier" and any(l.activation == "relu" for l in layers):
            recs.append("Xavier init is designed for tanh/sigmoid. For ReLU layers, kaimin is more appropriate.")
            
        if d.noise > 0.4 and d.samples < 2048:
            recs.append("High noise with few sampels is very challening. Increase samples or reduce noise.")
            
        if t.mixed_precision and t.device == "cpu":
            recs.append("Mixed precision on CPU has no benefit and may slow down training.")
            
        if t.early_stopping_patience is not None and t.early_stopping_min_delta == 0.0:
            recs.append("Early stopping is enabled with min_delta=0, any flat epoch will count. Set min_delta > 0.")
            
        return recs
        
    @staticmethod
    def _suggest_from_config(config: ProjectConfig) -> list[Suggestion]:
        suggestions: list[Suggestion] = []
        t = config.train
        m = config.model
        
        if t.optimizer == "sgd":
            suggestions.append(Suggestion(
                message="Switch from SGD to AdamW for more stable convergence.",
                config_patch=replace(config, train=replace(t, optimizer="adamw", learning_rate=1e-3, weight_decay=1e-4)),
            ))
            
        if t.scheduler == "none" and t.epochs >= 20:
            suggestions.append(Suggestion(
                message="Add a cosine LR scheduler for smoother convergence over long runs.",
                config_patch=replace(config, train=replace(t, scheduler="cosine")),
            ))
            
        if m.family == "mlp" and len(m.layers) >= 4 and t.gradient_clip_norm is None:
            suggestions.append(Suggestion(
                message="Deep MLP without gradient clipping. Add clip_norm=1.0 for stability.",
                config_patch=replace(config, train=replace(t, gradient_clip_norm=1.0)),
            ))
            
        if t.early_stopping_patience is None and t.epochs >= 30:
            suggestions.append(Suggestion(
                message="No early stopping configured. Add patience=10 to avoid wasted epochs.",
                config_patch=replace(config, train=replace(t, early_stopping_patience=10, early_stopping_min_delta=1e-4)),
            ))
            
        if m.initialization == "xavier" and any(l.activation == "relu" for l in m.layers):
            new_layers = [replace(l, activation="gelu") if l.activation == "relu" else l for l in m.layers]
            suggestions.append(Suggestion(
                message="ReLU with xavier init: switch activations to GELU or init to Kaiming.",
                config_patch=replace(config, model=replace(m, layers=new_layers)),
            ))
            
        return suggestions
    
    @staticmethod
    def _check_training(config: ProjectConfig, history: list[EpochMetrics], diagnostics: dict[str, object]) -> list[str]:
        recs: list[str] = []
        first = history[0]
        last = history[-1]
        is_clf = config.task == TaskType.binary_classification
        
        if last.val_loss >= first.val_loss * 0.98:
            recs.append("Validation loss barely improved. Train longer, tune LR, or use a larger model.")

        tail = history[-(max(1, len(history) // 5)):]
        if len(tail) >= 3:
            if max(e.val_loss for e in tail) - min(e.val_loss for e in tail) < 1e-3 and last.val_loss < first.val_loss * 0.98:
                recs.append("Val loss plateaued. Try LR decay, more capacity, or stop early.")

        if len(history) >= 6:
            deltas = [history[i].val_loss - history[i - 1].val_loss for i in range(1, len(history))]
            sign_flips = sum(1 for i in range(1, len(deltas)) if deltas[i] * deltas[i - 1] < 0)
            if sign_flips > len(deltas) * 0.6:
                recs.append("Val loss is oscillating. Lower LR or increase batch size.")

        gap = last.val_loss - last.train_loss
        if gap > 0.10:
            recs.append("Overfitting: val_loss >> train_loss. Increase dropout/weight_decay or reduce capacity.")
        if len(history) >= 4:
            early_gap = history[len(history) // 4].val_loss - history[len(history) // 4].train_loss
            if gap > early_gap * 1.5 and gap > 0.05:
                recs.append("Overfitting is worsening over time. Add early stopping or stronger regularisation.")

        if last.train_loss > first.train_loss * 0.95 and len(history) >= 5:
            recs.append("Train loss not decreasing. LR may be too low, or model too small.")
        if last.train_loss < 0.01:
            recs.append("Train loss near zero — possible severe overfitting or label leakage.")

        if is_clf:
            if last.val_accuracy < 0.70:
                recs.append("Low validation accuracy. Try more capacity, longer training, or less noise.")
            if 0.49 < last.val_accuracy < 0.52:
                recs.append("Accuracy near 50% — model may not be learning. Check class balance.")
        else:
            if last.val_accuracy < 0.30:
                recs.append("Low R² score. Check feature scaling and target distribution.")

        grad = diagnostics.get("avg_grad_norm", [])
        if isinstance(grad, list) and grad:
            grad_f = [float(x) for x in grad]
            if any(g > 50.0 for g in grad_f):
                recs.append("Large gradient norms. Lower LR or add gradient clipping.")
            if all(g < 1e-3 for g in grad_f[-(max(1, len(grad_f) // 5)):]):
                recs.append("Vanishing gradients. Try residual connections, GELU, or higher LR.")

        update = diagnostics.get("avg_weight_update_norm", [])
        if isinstance(update, list) and update:
            upd_f = [float(x) for x in update]
            if all(u < 1e-4 for u in upd_f):
                recs.append("Weight updates very small — LR may be too low.")
            if any(u > 1.0 for u in upd_f):
                recs.append("Weight updates very large — consider gradient clipping or lower LR.")

        dead = diagnostics.get("avg_dead_neuron_ratio", [])
        if isinstance(dead, list) and dead:
            dead_f = [float(x) for x in dead]
            mx = max(dead_f)
            if mx > 0.5:
                recs.append(f"Severe dead-neuron ratio ({mx:.0%}). Switch to GELU/LeakyReLU or lower dropout.")
            elif mx > 0.35:
                recs.append("Elevated dead-neuron ratio. Try GELU/LeakyReLU or lower dropout.")

        return recs

    @staticmethod
    def _suggest_from_training(config: ProjectConfig, history: list[EpochMetrics], diagnostics: dict[str, object]) -> list[Suggestion]:
        suggestions: list[Suggestion] = []
        t = config.train
        m = config.model
        first, last = history[0], history[-1]
        
        if last.val_loss - last.train_loss > 0.10:
            new_layers = [replace(l, dropout=min(0.4, l.dropout + 0.1)) for l in m.layers]
            suggestions.append(Suggestion(
                message="Overfitting detected: increase dropout by 0.1 and weight_decay.",
                config_patch=replace(
                    config,
                    model=replace(m, layers=new_layers),
                    train=replace(t, weight_decay=min(t.weight_decay * 5, 1e-2)),
                ),
            ))
            
        if last.train_loss > first.train_loss * 0.95 and len(history) >= 5:
            new_layers = [replace(l, units=min(l.units * 2, 512)) for l in m.layers]
            suggestions.append(Suggestion(
                message="Underfitting: double layer units (capped at 512) and raise LR slightly.",
                config_patch=replace(
                    config,
                    model=replace(m, layers=new_layers),
                    train=replace(t, learning_rate=min(t.learning_rate * 3, 1e-2)),
                ),
            ))
            
        dead = diagnostics.get("avg_dead_neuron_ratio", [])
        if isinstance(dead, list) and dead and max(float(x) for x in dead) > 0.35:
            new_layers = [replace(l, activation="gelu") if l.activation == "relu" else l for l in m.layers]
            if new_layers != m.layers:
                suggestions.append(Suggestion(
                    message="High dead-neuron ratio: switch ReLu activations to GELU.",
                    config_patch=replace(config, model=replace(m, layers=new_layers)),
                ))
                
        if len(history) >= 6:
            deltas = [history[i].val_loss - history[i-1].val_loss for i in range(1, len(history))]
            sign_flips = sum(1 for i in range(1, len(deltas)) if deltas[i] * deltas[i-1] < 0)
            if sign_flips > len(deltas) * 0.6:
                suggestions.append(Suggestion(
                    message="Osciallating loss: halve the learning rate.",
                    config_patch=replace(config, train=replace(t, learning_rate=t.learning_rate / 2)),
                ))
                
        tail = history[-(max(1, len(history) // 5)):]
        plateau = len(tail) >= 3 and max(e.val_loss for e in tail) - min(e.val_loss for e in tail) < 1e-3
        if plateau and t.scheduler == "none":
            suggestions.append(Suggestion(
                message="Plateau detected: add cosine LR decay to escape it.",
                config_patch=replace(config, train=replace(t, scheduler="cosine")),
            ))
            
        return suggestions