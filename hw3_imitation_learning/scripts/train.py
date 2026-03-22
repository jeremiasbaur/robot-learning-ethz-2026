"""Training script for SO-100 action-chunking imitation learning.

Imports a model from hw3.model and trains it on
state -> action-chunk prediction using the processed zarr dataset.

Usage:
    python scripts/train.py --zarr datasets/processed/single_cube/processed_ee_xyz.zarr \
        --state-keys ... \
        --action-keys ...
"""

# --state-keys state_ee_full "state_cube[:3]" state_joints state_obstacle --action-keys action_ee_xyz action_gripper
# --state-keys state_ee_xyz state_gripper "state_cube[:3]" state_obstacle --action-keys action_ee_xyz action_gripper
# --state-keys state_ee_xyz state_gripper "original_pos_cube_red[:3]" "original_pos_cube_green[:3]" "original_pos_cube_blue[:3]" state_goal goal_pos --action-keys action_ee_xyz action_gripper

# python scripts/train.py --zarr C:\Users\Jeremias\Documents\GitHub\robot-learning-ethz-2026\hw3_imitation_learning\datasets\processed\multi_cube\processed_ee_xyz_augmented.zarr --state-keys state_ee_xyz state_gripper "original_pos_cube_red[:3]" "original_pos_cube_green[:3]" "original_pos_cube_blue[:3]" state_goal goal_pos --action-keys action_ee_xyz action_gripper --policy multitask
# python scripts/train.py --zarr C:\Users\Jeremias\Documents\GitHub\robot-learning-ethz-2026\hw3_imitation_learning\datasets\processed\multi_cube\processed_ee_xyz_augmented.zarr --state-keys state_ee_xyz state_gripper "original_pos_cube_red[:3]" "original_pos_cube_green[:3]" "original_pos_cube_blue[:3]" state_goal goal_pos --action-keys action_ee_xyz action_gripper --policy multitask

# python scripts/compute_actions.py --augment-colors --action-space ee --datasets-dir C:\Users\Jeremias\Documents\GitHub\robot-learning-ethz-2026\hw3_imitation_learning\datasets\raw\multi_cube


from __future__ import annotations


import argparse
from pathlib import Path

import torch
import zarr as zarr_lib
from hw3.dataset import (
    Normalizer,
    SO100ChunkDataset,
    load_and_merge_zarrs,
    load_zarr,
)
#from hw3.selective_normalizer import SelectiveNormalizer

from hw3.model import BasePolicy, build_policy

# TODO: Any imports you want from torch or other libraries we use. Not allowed: libraries we don't use
from torch.utils.data import DataLoader, random_split

# TODO: Choose your own hyperparameters!
EPOCHS = 2000
BATCH_SIZE = 128
LR = 1e-4
VAL_SPLIT = 0.1


def train_one_epoch(
    model: BasePolicy,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        states, action_chunks = batch
        # TODO: Implement the training step for one batch here.
        # This mostly: Get states and action_chunks onto the correct device, compute the loss, and step the optimizer.
        optimizer.zero_grad()
        action_chunks = action_chunks.to(device)
        states = states.to(device)
        action_model = model(states)

        loss = torch.nn.functional.mse_loss(action_model, action_chunks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(
    model: BasePolicy,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    n_batches = 0
    
    goal_onehot = None

    for batch in loader:
        states, action_chunks = batch
        # TODO: Implement the evaluation step for one batch here.
        with torch.no_grad():
            states = states.to(device)
            action_model = model(states)
            action_chunks = action_chunks.to(device)

            loss = torch.nn.functional.mse_loss(action_model, action_chunks)
            total_loss += loss.item()
            n_batches += 1
        #goal_onehot = states[-1, :]#13:16]
        
    #Debug
    #attn = model.last_attn_weights[-1, 0].cpu().numpy()
    #print(f"Attention: red={attn[0]:.3f}, green={attn[1]:.3f}, blue={attn[2]:.3f}")
    #print(f"Goal: {goal_onehot}")
    return total_loss / max(n_batches, 1)


def main() -> None:
    # TODO: You may add any cli arguments that make life easier for you like learning rate etc.
    parser = argparse.ArgumentParser(description="Train action-chunking policy.")
    parser.add_argument(
        "--zarr", type=Path, required=True, help="Path to processed .zarr store."
    )
    parser.add_argument(
        "--policy",
        choices=["obstacle", "multitask"],
        default="obstacle",
        help="Policy type: 'obstacle' for single-cube obstacle scene, 'multitask' for multicube (default: obstacle).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=16,
        help="Action chunk horizon H (default: 16).",
    )
    parser.add_argument(
        "--state-keys",
        nargs="+",
        default=None,
        help='State array key specs to concatenate, e.g. state_ee_xyz state_gripper "state_cube[:3]". '
        "Supports column slicing with [:N], [M:], [M:N]. "
        "If omitted, uses the state_key attribute from the zarr metadata.",
    )
    parser.add_argument(
        "--action-keys",
        nargs="+",
        default=None,
        help="Action array key specs to concatenate, e.g. action_ee_xyz action_gripper. "
        "Supports column slicing with [:N], [M:], [M:N]. "
        "If omitted, uses the action_key attribute from the zarr metadata.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── load data ─────────────────────────────────────────────────────
    zarr_paths = [args.zarr]
    #if args.extra_zarr:
    #   zarr_paths.extend(args.extra_zarr)

    if len(zarr_paths) == 1:
        states, actions, ep_ends = load_zarr(
            args.zarr,
            state_keys=args.state_keys,
            action_keys=args.action_keys,
        )
    else:
        print(f"Merging {len(zarr_paths)} zarr stores: {[str(p) for p in zarr_paths]}")
        states, actions, ep_ends = load_and_merge_zarrs(
            zarr_paths,
            state_keys=args.state_keys,
            action_keys=args.action_keys,
        )
    # Determine which state indices to skip normalization for
    # For multicube: state_goal is at indices [13:16] based on provided state_keys in above python train command
    # state_ee_xyz(3) + state_gripper(1) + red(3) + green(3) + blue(3) + goal(3) + goal_pos(3)
    skip_indices = None
    if args.policy == "multitask" and args.state_keys:
        # Calculate where state_goal is in the concatenated state
        # This assumes your state_keys order from line 14
        state_dim_so_far = 0
        for key in args.state_keys:
            if "state_goal" in key:
                # state_goal is 3-dimensional, skip these indices
                skip_indices = {state_dim_so_far, state_dim_so_far + 1, state_dim_so_far + 2}
                print(f"Skipping normalization for state_goal at indices {skip_indices}")
                break
            # Count dimensions for this key
            if "[:3]" in key or "state_ee_xyz" in key or "goal_pos" in key:
                state_dim_so_far += 3
            elif "state_gripper" in key:
                state_dim_so_far += 1
            elif "state_goal" in key:
                state_dim_so_far += 3
            # Add more cases if needed
    
    # if skip_indices:
    #     normalizer = SelectiveNormalizer.from_data(states, actions, skip_state_indices=skip_indices)
    # else:
    #     normalizer = Normalizer.from_data(states, actions)
    
    normalizer = Normalizer.from_data(states, actions)
    dataset = SO100ChunkDataset(
        states,
        actions,
        ep_ends,
        chunk_size=args.chunk_size,
        normalizer=normalizer,
    )
    print(f"Dataset: {len(dataset)} samples, chunk_size={args.chunk_size}")
    print(f"  state_dim={states.shape[1]}, action_dim={actions.shape[1]}")

    # ── train / val split ─────────────────────────────────────────────
    n_val = max(1, int(len(dataset) * VAL_SPLIT))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed)
    )

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    # ── model ─────────────────────────────────────────────────────────
    model = build_policy(
        args.policy,
        state_dim=states.shape[1],
        action_dim=actions.shape[1],
        chunk_size=args.chunk_size,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    # TODO: implement an optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS-100, eta_min=2e-5)
    warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=100)
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [warmup, cosine], milestones=[100])
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    # ── training loop ─────────────────────────────────────────────────
    best_val = float("inf")

    # Derive action space tag from action keys (e.g. "ee_xyz", "joints")
    action_space = "unknown"
    if args.action_keys:
        for k in args.action_keys:
            base = k.split("[")[0]  # strip column slices
            if base != "action_gripper":
                action_space = base.removeprefix("action_")
                break

    save_name = f"best_model_{action_space}_{args.policy}_32.pt"

    n_dagger_eps = 0
    for zp in zarr_paths:
        z = zarr_lib.open_group(str(zp), mode="r")
        n_dagger_eps += z.attrs.get("num_dagger_episodes", 0)
    if n_dagger_eps > 0:
        save_name = f"best_model_{action_space}_{args.policy}_dagger{n_dagger_eps}ep.pt"
    # Default: checkpoints/<task>/
    if "multi_cube" in str(args.zarr):
        ckpt_dir = Path("./checkpoints/multi_cube")
    else:
        ckpt_dir = Path("./checkpoints/single_cube")
    save_path = ckpt_dir / save_name
    save_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss = evaluate(model, val_loader, device)
        scheduler.step()

        tag = ""
        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "normalizer": {
                        "state_mean": normalizer.state_mean,
                        "state_std": normalizer.state_std,
                        "action_mean": normalizer.action_mean,
                        "action_std": normalizer.action_std,
                    },
                    "chunk_size": args.chunk_size,
                    "policy_type": args.policy,
                    "state_keys": args.state_keys,
                    "action_keys": args.action_keys,
                    "state_dim": int(states.shape[1]),
                    "action_dim": int(actions.shape[1]),
                    "val_loss": val_loss,
                },
                save_path,
            )
            tag = " ✓ saved"

        print(
            f"Epoch {epoch:3d}/{EPOCHS} | "
            f"train {train_loss:.6f} | val {val_loss:.6f}{tag}"
        )

    print(f"\nBest val loss: {best_val:.6f}")
    print(f"Checkpoint: {save_path}")


if __name__ == "__main__":
    main()
