import torch
from rwkv7_ttt_phantom import create_model, RWKV7PhantomConfig
from Training import load_pretrained_rwkv7


def small_config(**kwargs):
    params = dict(n_layer=1, n_embd=8, n_head=2, head_size=4, vocab_size=16, core_ratio=0.5, ghost_ratio=0.5, ttt_enabled=False)
    params.update(kwargs)
    return RWKV7PhantomConfig(**params)


def test_create_model_forward():
    config = small_config()
    model = create_model(config)
    x = torch.randint(0, config.vocab_size, (2, 3))
    logits, state = model(x)
    assert logits.shape == (2, 3, config.vocab_size)
    assert state is not None


def test_init_state_shapes():
    config = small_config()
    model = create_model(config)
    batch_size = 4
    state = model.init_state(batch_size)
    assert len(state) == config.n_layer * 3
    assert state[0].shape == (batch_size, 1, config.n_embd)
    assert state[1].shape == (batch_size, 1, config.n_embd)
    assert state[2].shape == (
        batch_size,
        config.n_head,
        config.head_size,
        config.head_size,
    )


def test_load_pretrained_rwkv7(tmp_path):
    config = small_config()
    ckpt = {
        "blocks.0.att.receptance.weight": torch.randn(config.n_embd, config.n_embd)
    }
    ckpt_path = tmp_path / "mock.pth"
    torch.save(ckpt, ckpt_path)
    model = load_pretrained_rwkv7(str(ckpt_path), config)
    x = torch.randint(0, config.vocab_size, (1, 2))
    logits, _ = model(x)
    assert logits.shape == (1, 2, config.vocab_size)
