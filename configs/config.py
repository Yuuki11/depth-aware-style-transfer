"""Training configuration with sensible defaults."""

from dataclasses import dataclass, field


@dataclass
class TrainConfig:
    """All training hyperparameters in one place."""

    # Paths
    content_dir: str = "./data/coco/train2017"
    style_dir: str = "./data/wikiart"
    checkpoint_dir: str = "./checkpoints"
    sample_dir: str = "./samples"

    # Dataset
    max_content_images: int | None = None  # None = use all
    max_style_images: int | None = None
    image_size: int = 256

    # Training
    epochs: int = 50
    batch_size: int = 32
    num_workers: int = 4

    # Loss weights
    content_weight: float = 1.2
    style_weight: float = 25.0

    # Optimizer
    lr_encoder_low: float = 1e-5   # conv1, layer1, layer2
    lr_encoder_high: float = 1e-4  # layer3, layer4
    lr_decoder: float = 1e-3
    weight_decay: float = 5e-5
    grad_clip: float = 1.0

    # Scheduler
    scheduler: str = "cosine"  # "cosine" or "step"
    T_max: int = 50

    # Depth (for depth-aware training)
    use_depth: bool = False
    depth_scale: float = 2.0

    # Logging
    log_interval: int = 100  # Print every N batches
    save_interval: int = 1   # Save checkpoint every N epochs
    sample_interval: int = 1 # Generate samples every N epochs

    # Pretrained
    pretrained_encoder: bool = False
    resume_from: str | None = None  # Path to checkpoint to resume from
