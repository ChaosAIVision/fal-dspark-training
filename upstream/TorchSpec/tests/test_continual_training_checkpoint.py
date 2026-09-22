from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from torchspec.training import checkpoint


class DummyBF16Optimizer:
    def __init__(self, *, lr: float, betas: tuple[float, float], weight_decay: float) -> None:
        self.fp32_params = [torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float32))]
        self.optimizer = torch.optim.AdamW(
            self.fp32_params,
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
        )

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)

    def sync_fp32_params_from_model(self):
        raise AssertionError("fallback path should not run in this test")


def test_restore_fp32_master_params_keeps_fresh_optimizer_hparams():
    optimizer = DummyBF16Optimizer(lr=0.123, betas=(0.7, 0.8), weight_decay=0.456)
    actor = SimpleNamespace(model=mock.MagicMock(), optimizer=optimizer)

    checkpoint_optimizer = DummyBF16Optimizer(lr=0.999, betas=(0.1, 0.2), weight_decay=0.0)
    with torch.no_grad():
        checkpoint_optimizer.fp32_params[0].copy_(torch.tensor([42.0], dtype=torch.float32))
    checkpoint_optimizer.optimizer.state[checkpoint_optimizer.fp32_params[0]]["step"] = (
        torch.tensor(7.0)
    )

    checkpoint_state = {
        "optim": checkpoint_optimizer.optimizer.state_dict(),
        "fp32_params": {"0": checkpoint_optimizer.fp32_params[0].detach().clone()},
    }

    def fake_dcp_load(*, state_dict, checkpoint_id):
        assert checkpoint_id == "/tmp/fake_optim"
        state_dict["optim_state"].load_state_dict(checkpoint_state)

    def fake_exists(self):
        return str(self) in {"/tmp/fake_optim", "/tmp/fake_optim/.metadata"}

    with (
        mock.patch("torchspec.training.checkpoint.dcp.load", side_effect=fake_dcp_load),
        mock.patch("pathlib.Path.exists", new=fake_exists),
    ):
        checkpoint._restore_fp32_master_params(actor, Path("/tmp/fake_optim"))

    group = optimizer.optimizer.param_groups[0]
    assert group["lr"] == 0.123
    assert group["betas"] == (0.7, 0.8)
    assert group["weight_decay"] == 0.456
    assert optimizer.optimizer.state == {}
    assert torch.equal(optimizer.fp32_params[0].detach(), torch.tensor([42.0], dtype=torch.float32))


def test_optimizer_only_resume_loads_optimizer_but_not_scheduler_or_progress(tmp_path):
    actor = SimpleNamespace(
        args=SimpleNamespace(
            load_path=str(tmp_path),
            ckpt_step=None,
            continual_training=True,
            resume_optimizer_state_only=True,
            start_step=None,
        ),
        model=object(),
        optimizer=object(),
        lr_scheduler=object(),
        global_step=0,
    )
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("7")
    model_dir = tmp_path / "iter_0000007" / "model"
    optimizer_dir = tmp_path / "iter_0000007" / "optimizer"
    scheduler_dir = tmp_path / "iter_0000007" / "lr_scheduler"
    for path in (model_dir, optimizer_dir, scheduler_dir):
        path.mkdir(parents=True)
    (tmp_path / "iter_0000007" / "meta.json").write_text(
        '{"global_step": 7, "next_step": 7}'
    )

    with (
        mock.patch.object(checkpoint, "ModelState", return_value="model-state"),
        mock.patch.object(checkpoint, "OptimizerState", return_value="optimizer-state"),
        mock.patch.object(checkpoint, "LRSchedulerState", return_value="scheduler-state"),
        mock.patch.object(checkpoint.dcp, "load") as dcp_load,
        mock.patch.object(checkpoint.dist, "barrier"),
        mock.patch.object(checkpoint.torch.cuda, "synchronize"),
    ):
        payload = checkpoint.load(actor)
        checkpoint.finalize_load(actor, payload)

    loaded_ids = [call.kwargs["checkpoint_id"] for call in dcp_load.call_args_list]
    assert str(model_dir) in loaded_ids
    assert str(optimizer_dir) in loaded_ids
    assert str(scheduler_dir) not in loaded_ids
    assert actor.global_step == 0
    assert actor.args.start_step is None
