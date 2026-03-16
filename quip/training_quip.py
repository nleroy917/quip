"""
Quip — Training utilities and Trainer subclass.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from transformers import Trainer, TrainingArguments


@dataclass
class QuipTrainingArguments(TrainingArguments):
    """
    Extends HF TrainingArguments with Quip-specific defaults.
    """

    # quip defaults from the research plan
    freeze_backbone: bool = field(
        default=False,
        metadata={"help": "Freeze CLIP backbone and only train quant heads (warm-up phase)."},
    )
    freeze_backbone_steps: int = field(
        default=0,
        metadata={"help": "Number of steps to freeze backbone before unfreezing. 0 = use freeze_backbone flag."},
    )


class QuipTrainer(Trainer):
    """
    Trainer subclass that handles Quip-specific training logic.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._backbone_frozen = False
        if self.args.freeze_backbone or self.args.freeze_backbone_steps > 0:
            self._set_backbone_grad(False)

    def _set_backbone_grad(self, requires_grad: bool):
        for param in self.model.clip.parameters():
            param.requires_grad = requires_grad
        self._backbone_frozen = not requires_grad

    def training_step(self, model, inputs, num_items_in_batch=None):
        if (
            self._backbone_frozen
            and self.args.freeze_backbone_steps > 0
            and self.state.global_step >= self.args.freeze_backbone_steps
        ):
            self._set_backbone_grad(True)
            self.create_optimizer()

        return super().training_step(model, inputs, num_items_in_batch=num_items_in_batch)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(
            pixel_values=inputs.get("pixel_values"),
            input_ids=inputs.get("input_ids"),
            attention_mask=inputs.get("attention_mask"),
            return_loss=True,
        )
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss