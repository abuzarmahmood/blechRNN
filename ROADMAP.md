# BlechRNN Roadmap

This document tracks planned improvements, open design questions, and the reasoning
behind key architectural decisions for the project.

---

## Potential Migration to PyTorch Lightning

### Background

The current implementation uses plain PyTorch.  The training logic lives in
`src/train.py` and includes a hand-written training loop, manual device
placement, early stopping, and model checkpointing.  The question is whether
migrating to [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/) would
improve the project enough to justify the effort.

### Current Implementation Summary

| Concern | Where it lives today |
|---|---|
| Model definition | `src/model.py` – `nn.Module` subclasses (`CTRNN`, `CTRNN_plus_output`, `autoencoderRNN`) |
| Training loop | `src/train.py` – `train_model()` function |
| Device management | Manual `torch.device` calls in `src/run_model.py` and `src/train.py` |
| Optimizer | Hard-coded `optim.Adam` in `train_model()` |
| Checkpointing | Manual `torch.save(net.state_dict(), save_path)` inside the loop |
| Early stopping | Manual `patience_counter` logic inside the loop |
| Cross-validation loss | Computed every 100 steps with an inline block |
| Loss functions | `smooth_MSELoss`, `MSELoss` wrappers in `src/train.py` |

### Pros of Switching to PyTorch Lightning

1. **Less boilerplate.**  Lightning's `LightningModule` consolidates
   `training_step`, `validation_step`, `configure_optimizers`, and logging into
   a single class.  The repetitive gradient-zeroing / loss-backward / optimizer-
   step pattern in `train_model()` disappears.

2. **Built-in callbacks.**  Early stopping (`EarlyStopping`) and model
   checkpointing (`ModelCheckpoint`) are first-class callbacks.  Replacing the
   current `patience_counter` and `torch.save` blocks with two callback
   declarations reduces error-prone hand-written logic.

3. **Hardware abstraction.**  The `Trainer` object handles CPU, single-GPU, and
   multi-GPU/TPU transparently.  The current manual `.to(device)` calls
   scattered across `run_model.py` would no longer be necessary.

4. **Reproducible logging.**  Lightning integrates with TensorBoard, W&B, and
   CSV loggers out of the box.  Replacing the `print` statements and manual
   `loss_history` list with `self.log(...)` calls gives persistent, comparable
   experiment records with minimal code.

5. **Separation of concerns.**  Research code (model + loss) is cleanly
   separated from engineering code (distributed strategy, precision, profiling).
   This matters as the project grows or is used across multiple compute
   environments.

6. **Built-in gradient clipping, mixed precision, and profiling.**  These are
   single `Trainer` arguments rather than custom code, reducing maintenance
   burden.

7. **Easier testing.**  `LightningModule` instances can be unit-tested without
   instantiating a full `Trainer`, making it straightforward to write regression
   tests for the forward pass and loss calculation.

### Cons of Switching to PyTorch Lightning

1. **New dependency.**  `pytorch-lightning` (or `lightning`) is not currently in
   `requirements.txt`.  Adding it pins another large package with its own release
   cadence and potential conflicts with the pinned `torch==2.3.1`.

2. **Learning curve.**  Contributors familiar with plain PyTorch need to learn
   Lightning's conventions (`LightningModule`, `LightningDataModule`, `Trainer`
   flags).  For a small project, this overhead may outweigh the benefits.

3. **Abstraction overhead for simple loops.**  The current training loop in
   `train_model()` is roughly 50 lines and is easy to read and debug.  Lightning
   adds indirection (hooks, stages, `self.log`) that can obscure what is actually
   happening during training.

4. **Less control over the training loop.**  The current code computes
   cross-validation loss every 100 steps and breaks early based on a patience
   counter.  While Lightning supports custom logic through `on_train_batch_end`
   and `on_validation_epoch_end` hooks, achieving exactly the same behaviour
   requires understanding those hooks.

5. **Migration cost.**  Each `nn.Module` would need to be wrapped in or replaced
   by a `LightningModule`.  `run_model.py` would need significant refactoring.
   Existing notebooks or downstream scripts that import `train_model` directly
   would break.

6. **Version stability risk.**  PyTorch Lightning has a history of significant
   API changes between major versions (e.g., the rename from
   `pytorch_lightning` to `lightning.pytorch`).  Adopting it locks the project
   into tracking those changes.

### Recommendation

**Defer migration; address incrementally.**

The current codebase is small and the training loop is straightforward.  The
marginal benefit of Lightning does not yet justify the migration cost or the
additional dependency.  However, several Lightning-inspired improvements can be
made to the existing plain-PyTorch code at low cost:

- [ ] Extract device handling into a shared utility so `.to(device)` is not
  repeated in every script.
- [ ] Move `EarlyStopping` and `ModelCheckpoint` logic into reusable helper
  classes inside `train.py`, mirroring Lightning's callback API.
- [ ] Replace `print`-based logging with Python's `logging` module, making it
  easy to redirect output to a file or external logger later.
- [ ] Add a `DataLoader`-based data pipeline in `get_data.py` so a future
  migration to `LightningDataModule` would be a thin wrapper.

Revisit this decision if any of the following occur:

- The project needs to scale to multi-GPU or distributed training.
- Experiment tracking and reproducibility become a bottleneck.
- The training loop grows significantly in complexity (e.g., curriculum
  learning, adversarial training, multiple optimizers).

---

## Other Planned Improvements

- [ ] Add unit tests for `model.py` and `train.py`.
- [ ] Publish a versioned dataset loader so external users can reproduce
  results without the internal `ephys_data` dependency in `run_model.py`.
- [ ] Add a configuration file (e.g., YAML or TOML) so hyperparameters do not
  need to be edited directly in source files.
- [ ] Document the expected `.h5` file schema in `get_data.py`.
