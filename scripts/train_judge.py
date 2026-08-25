"""Train the independent judge likelihood model.

The judge scores how realistic a generated scenario is. Scoring SMART samples
with SMART would be circular, so the judge differs from the generator in
width and depth, uses its own seed, and is validated on scenarios the
generator never trained on.

The architecture check is enforced at startup: a judge that silently matches
the generator produces numbers that look fine and mean nothing.

Local smoke:
    python scripts/train_judge.py --config configs/judge/judge_local_smoke.yaml \
        --save_ckpt_path checkpoints/judge_smoke

Full run (8x A800):
    python scripts/train_judge.py --config configs/judge/judge_server.yaml \
        --save_ckpt_path checkpoints/judge
"""
import os
import sys
from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
from torch_geometric.loader import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from smart.datasets.scalable_dataset import MultiDataset
from smart.model import SMART
from smart.safety.judge import (assert_distinct_architecture,
                                generator_architecture_from_checkpoint,
                                judge_architecture_from_config,
                                scenario_ids_for)
from smart.transforms import WaymoTargetBuilder
from smart.utils.config import load_config_act


def build_dataset(config, split, raw_dir, split_name):
    """Dataset for one split, restricted by the config's partition if named."""
    scenario_ids = None
    if split_name:
        scenario_ids = scenario_ids_for(config, split_name, raw_dir)
        if not scenario_ids:
            raise ValueError(f'split {split_name!r} selected no scenarios from {raw_dir}')
    return MultiDataset(
        root=config.Dataset.root,
        split=split,
        raw_dir=raw_dir,
        processed_dir=None,
        transform=WaymoTargetBuilder(config.Model.num_historical_steps,
                                     config.Model.decoder.num_future_steps),
        scenario_ids=scenario_ids,
    )


def main():
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/judge/judge_local_smoke.yaml')
    parser.add_argument('--generator_ckpt', type=str, default='checkpoints/epoch=31.ckpt')
    parser.add_argument('--save_ckpt_path', type=str, default='checkpoints/judge')
    args = parser.parse_args()

    config = load_config_act(args.config)
    pl.seed_everything(config.Trainer.seed, workers=True)

    # Refuse to train a judge that cannot be told apart from the generator.
    assert_distinct_architecture(
        judge_architecture_from_config(config),
        generator_architecture_from_checkpoint(args.generator_ckpt))

    data_config = config.Dataset
    train_ds = build_dataset(config, 'train', data_config.train_raw_dir,
                             config.Split.train_split)
    val_ds = build_dataset(config, 'val', data_config.val_raw_dir,
                           config.Split.val_split)
    print(f'judge train scenarios: {len(train_ds)} | val scenarios: {len(val_ds)}')

    loader = dict(num_workers=data_config.num_workers,
                  pin_memory=data_config.pin_memory,
                  persistent_workers=data_config.num_workers > 0)
    train_loader = DataLoader(train_ds, batch_size=data_config.train_batch_size,
                              shuffle=True, **loader)
    val_loader = DataLoader(val_ds, batch_size=data_config.val_batch_size,
                            shuffle=False, **loader)

    trainer_config = config.Trainer
    trainer = pl.Trainer(
        accelerator=trainer_config.accelerator,
        devices=trainer_config.devices,
        strategy=DDPStrategy(find_unused_parameters=True, gradient_as_bucket_view=True)
        if trainer_config.devices > 1 else 'auto',
        num_nodes=trainer_config.num_nodes,
        accumulate_grad_batches=trainer_config.accumulate_grad_batches,
        max_epochs=trainer_config.max_epochs,
        callbacks=[
            ModelCheckpoint(dirpath=args.save_ckpt_path, filename='judge-{epoch:02d}',
                            monitor='val_cls_acc', mode='max', save_top_k=3,
                            every_n_epochs=1),
            LearningRateMonitor(logging_interval='epoch'),
        ],
        num_sanity_val_steps=0,
        gradient_clip_val=0.5,
    )
    trainer.fit(SMART(config.Model), train_loader, val_loader)


if __name__ == '__main__':
    main()
