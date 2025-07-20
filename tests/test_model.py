import sys
from pathlib import Path
import torch
import types  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rwkv7_ttt_phantom import RWKV7PhantomConfig, create_model  # noqa: E402

dummy_wandb = types.ModuleType("wandb")
dummy_wandb.sdk = types.SimpleNamespace(lib=types.SimpleNamespace(RunDisabled=None))
sys.modules.setdefault("wandb", dummy_wandb)
pl_wandb = types.ModuleType("pytorch_lightning.loggers.wandb")
pl_wandb.WandbLogger = object
sys.modules.setdefault("pytorch_lightning.loggers.wandb", pl_wandb)

from Training import load_pretrained_rwkv7  # noqa: E402


def _small_config():
    return RWKV7PhantomConfig(
        n_layer=1,
        n_embd=2,
        n_head=1,
        head_size=2,
        vocab_size=50,
        ttt_enabled=False,
        use_cuda_kernel=False,
    )


def test_create_model_forward():
    config = _small_config()
    model = create_model(config)
    x = torch.randint(0, config.vocab_size, (2, 2))
    logits, state = model(x)
    assert logits.shape == (2, 2, config.vocab_size)
    assert state is not None


def test_init_state_shapes():
    config = _small_config()
    model = create_model(config)
    batch_size = 3
    state = model.init_state(batch_size)
    assert len(state) == config.n_layer * 3
    assert state[0].shape == (batch_size, 1, config.n_embd)
    assert state[2].shape == (
        batch_size,
        config.n_head,
        config.head_size,
        config.head_size,
    )


def test_load_pretrained_rwkv7(tmp_path):
    config = _small_config()
    model = create_model(config)
    ckpt = tmp_path / "model.pth"
    torch.save(model.state_dict(), ckpt)
    loaded = load_pretrained_rwkv7(str(ckpt), config)
    x = torch.randint(0, config.vocab_size, (1, 2))
    logits, _ = loaded(x)
    assert logits.shape == (1, 2, config.vocab_size)
