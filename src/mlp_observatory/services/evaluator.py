from __future__ import annotations

from mlp_observatory.domain.models import EpochMetrics, ProjectConfig, TaskType


class RunEvaluator:
    @staticmethod
    def evaluate(config: ProjectConfig, history: list[EpochMetrics], diagnostics: dict[str, object]) -> list[str]:
        if not history:
            return ["No history collected. Check training pipeline."]

        recs: list[str] = []
        first = history[0]
        last = history[-1]

        if config.task == TaskType.regression:
            if last.val_loss > first.val_loss * 0.98:
                recs.append("Validation loss barely improved. Train longer, tune LR, or use a larger model.")
            if last.val_accuracy < 0.30:
                recs.append("Low regression score detected. Check feature scaling and target quality.")
        else:
            if last.val_loss > first.val_loss * 0.98:
                recs.append("Validation loss barely improved. Increase model capacity or train for more epochs.")
            if (last.train_loss + 0.08) < last.val_loss:
                recs.append("Potential overfitting. Increase dropout or weight decay, or reduce hidden layers.")
            if last.val_accuracy < 0.70:
                recs.append("Validation accuracy is low. Try larger hidden_dim or lower data noise.")

        dead = diagnostics.get("avg_dead_neuron_ratio", [])
        if isinstance(dead, list) and any(float(x) > 0.35 for x in dead):
            recs.append("High dead-neuron ratio detected. Try GELU/LeakyReLU, lower dropout, or lower LR.")

        grad = diagnostics.get("avg_grad_norm", [])
        if isinstance(grad, list) and any(float(x) > 50.0 for x in grad):
            recs.append("Large gradient norms detected. Lower learning rate or apply gradient clipping.")

        if config.model.family == "mlp":
            layer_count = len(config.model.layers)
            max_units = max((layer.units for layer in config.model.layers), default=0)
            if layer_count >= 6 and max_units >= 512:
                recs.append("Architecture is heavy for interactive runs. Consider smaller preset for faster iteration.")

        if not recs:
            recs.append("Training looks healthy. Try increasing epochs for potentially better validation score.")

        return recs
