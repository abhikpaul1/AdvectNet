# Models

Place trained checkpoints here locally.

Git ignores `*.pth`, `*.pt`, and `*.ckpt` files to avoid committing large binary artifacts.

Confirmed frozen-RAFT checkpoint format:

```python
{
    "unet": unet.state_dict(),
    "config": {
        "bt_min": 180.0,
        "bt_max": 310.0,
        "patch": 256,
    },
}
```

Experimental RAFT-fine-tuned checkpoints may additionally include:

```python
{
    "raft": raft.state_dict(),
    "unet": unet.state_dict(),
}
```

Do not present RAFT fine-tuning as the final architecture until its validation has been confirmed.
