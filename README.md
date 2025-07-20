# RWKV-7 TTT Phantom

RWKV-7 TTT Phantom is an experimental extension of the [RWKV](https://github.com/BlinkDL/RWKV-LM) family of models. It combines

* **RWKV‑7** – a linear‑time, attention‑free architecture that matches Transformer quality while keeping constant memory.
* **Test‑Time Training (TTT)** – a small gradient update performed on a hidden state during inference to improve predictions.
* **GhostRNN** – light‑weight RNN layers that split their weights into a small "core" (fully trained) and a cheap "ghost" projection.

The codebase demonstrates how these ideas can be integrated to build and train new RWKV models.

## Key Features

- Implementation of a configurable `RWKV7TTTPhantom` model in [rwkv7_ttt_phantom.py](rwkv7_ttt_phantom.py).
- PyTorch Lightning training pipeline with optional meta‑learning and weight compression in [Training.py](Training.py).
- Support for SVD‑based compression and structured sparsity.
- Autoregressive text generation and utility functions for loading pretrained RWKV‑7 checkpoints.

The project is licensed under the Apache‑2.0 license (see [LICENSE](LICENSE)).

## Requirements

- Python 3.9+
- PyTorch 2.x
- `pytorch-lightning==1.9.5`
- NumPy
- Weights & Biases (optional, used for logging)

Install dependencies with:

```bash
pip install torch pytorch-lightning==1.9.5 numpy wandb
```

## Usage

Create a model with the default configuration:

```python
from rwkv7_ttt_phantom import create_model
model = create_model()
```

For training, see the `train_model` function in [Training.py](Training.py). A minimal example is included at the bottom of that file. Training requires a tokenized dataset in `.bin/.idx` format (as used in the official RWKV codebase).

To generate text with a trained model:

```python
prompt = torch.tensor([[1, 2, 3]])
model.generate(prompt, max_new_tokens=20)
```

## Relationship to Official RWKV Models

RWKV‑7 TTT Phantom builds upon ideas from the official RWKV project. The official RWKV‑7 models are RNN‑based LLMs that achieve Transformer‑level performance without attention. They are trained with carefully tuned initialization and optimization strategies, and the code can be found on the [RWKV‑LM GitHub](https://github.com/BlinkDL/RWKV-LM).

This repository experiments with additional features:

- Test‑time gradient updates on a subset of the state (TTT) to adapt on the fly.
- GhostRNN layers that reduce computation by keeping a small core set of parameters.
- Optional compression (SVD and sparsity) after pretraining.

Because these changes modify the standard RWKV‑7 architecture, results may differ from the official checkpoints available from [rwkv.com](https://www.rwkv.com/). Use this project as a starting point for research rather than a drop‑in replacement.

## References

- [RWKV‑LM Repository](https://github.com/BlinkDL/RWKV-LM) – official RWKV implementation and training scripts.
- [RWKV Discord](https://discord.gg/bDSBUMeFpc) – community discussions and support.

## Disclaimer

This is experimental research code. It has not been tested at scale and is **not** an official RWKV release. Use it at your own risk.
