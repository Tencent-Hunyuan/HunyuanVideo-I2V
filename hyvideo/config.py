import argparse
from .constants import *
import re
from .modules.models import HUNYUAN_VIDEO_CONFIG


def parse_args(mode="eval", namespace=None):
    parser = argparse.ArgumentParser(description="HunyuanVideo inference/lora training script")

    parser = add_network_args(parser)
    parser = add_extra_models_args(parser)
    parser = add_denoise_schedule_args(parser)
    parser = add_i2v_args(parser)
    parser = add_lora_args(parser)
    parser = add_inference_args(parser)
    parser = add_parallel_args(parser)
    if mode == "train":
        parser = add_training_args(parser)
        parser = add_optimizer_args(parser)
        parser = add_deepspeed_args(parser)
        parser = add_data_args(parser)
        parser = add_train_denoise_schedule_args(parser)

    args = parser.parse_args(namespace=namespace)
    args = sanity_check_args(args)

    return args

def add_train_denoise_schedule_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group(title="Denoise schedule")

    group.add_argument("--flow-path-type", type=str, default="linear", choices=FLOW_PATH_TYPE,
                       help="Path type for flow matching schedulers.")
    group.add_argument("--flow-predict-type", type=str, default="velocity", choices=FLOW_PREDICT_TYPE,
                       help="Prediction type for flow matching schedulers.")
    group.add_argument("--flow-loss-weight", type=str, default=None, choices=FLOW_LOSS_WEIGHT,
                       help="Loss weight type for flow matching schedulers.")
    group.add_argument("--flow-train-eps", type=float, default=None,
                       help="Small epsilon for avoiding instability during training.")
    group.add_argument("--flow-sample-eps", type=float, default=None,
                       help="Small epsilon for avoiding instability during sampling.")
    group.add_argument("--flow-snr-type", type=str, default="lognorm", choices=FLOW_SNR_TYPE,
                       help="Type of SNR to use for flow matching schedulers.")

    return parser

def add_deepspeed_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group(title="DeepSpeed")

    group.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training.")
    group.add_argument("--zero-stage", type=int, default=0, choices=[0, 1, 2, 3],
                       help="DeepSpeed ZeRO stage. 0: off, 1: offload optimizer, 2: offload parameters, "
                            "3: offload optimizer and parameters.")
    return parser

def add_data_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group(title="Data")

    group.add_argument("--data-type", type=str, default="image", choices=DATA_TYPE, help="Type of the dataset.")
    group.add_argument("--data-jsons-path", type=str, default=None, help="Dataset path for training.")
    group.add_argument("--sample-n-frames", type=int, default=65,
                       help="How many frames to sample from a video. if using 3d vae, the number should be 4n+1")
    group.add_argument("--sample-stride", type=int, default=1,
                       help="How many frames to skip when sampling from a video.")
    group.add_argument("--num-workers", type=int, default=4, help="Number of workers for data loading.")
    group.add_argument("--prefetch-factor", type=int, default=2, help="Prefetch factor for data loading.")
    group.add_argument("--same-data-batch", action="store_true", help="Use same data type for all rank in a batch for training.")
    group.add_argument("--uncond-p", type=float, default=0.1,
                       help="Probability of randomly dropping video description.")
    group.add_argument("--sematic-cond-drop-p", type=float, default=0.1,
                       help="Probability of randomly dropping img condition description.")

    return parser

def add_training_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group(title="Training")

    group.add_argument("--task-flag", type=str, required=True,
                       help="Task flag for training/inference. It is used to determine the experiment directory.")
    group.add_argument("--output-dir", type=str, required=True, help="Directory to save logs and models")
    group.add_argument("--sample-dir", type=str, default=None, required=False, help="Directory to save samples")
    group.add_argument("--micro-batch-size", type=int, default=1, nargs='*',
                       help="Batch size per model instance (local batch size).")
    group.add_argument("--video-micro-batch-size", type=int, default=None, nargs='*',
                       help="Batch size per model instance (local batch size).")
    group.add_argument("--global-batch-size", type=int, default=None, nargs='*',
                       help="Global batch size (across all model instances). "
                            "global-batch-size = micro-batch-size * world-size * gradient-accumulation-steps")
    group.add_argument("--gradient-accumulation-steps", type=int, default=1,
                       help="Number of steps to accumulate gradients over before performing an update.")
    group.add_argument("--global-seed", type=int, default=42, help="Global seed for reproducibility.")

    group.add_argument("--resume", type=str, default=None,
                       help="Path to the checkpoint to resume training. It can be an experiment index to resume from "
                            "the latest checkpoint in the output directory.")
    group.add_argument("--init-from", type=str, default=None,
                       help="Path to the checkpoint to load from init ckpt for training. ")
    group.add_argument("--training-parts", type=str, default=None, help="Training a subset of the model parameters.")
    group.add_argument("--init-save", action="store_true", help="Save the initial model before training.")
    group.set_defaults(final_save=True)
    group.add_argument("--final-save", action="store_true", help="Save the final model after training.")
    group.add_argument("--no-final-save", dest="final_save", action="store_false", help="Do not save the final model.")

    group.add_argument("--epochs", type=int, default=100, help="Number of epochs to train.")
    group.add_argument("--max-training-steps", type=int, default=10_000_000, help="Maximum number of training steps.")
    group.add_argument("--ckpt-every", type=int, default=5000, help="Save checkpoint every N steps.")

    group.add_argument("--rope-theta-rescale-factor", type=float, default=1.0, nargs='+',
                       help="Rope interpolation factor.")
    group.add_argument("--rope-interpolation-factor", type=float, default=1.0, nargs='+',
                       help="Rope interpolation factor.")

    group.add_argument("--log-every", type=int, default=10, help="Log every N update steps.")
    group.add_argument("--tensorboard", action="store_true", help="Enable TensorBoard logging.")
    group.add_argument("--profile", action="store_true", help="Enable PyTorch profiler.")
    return parser

def add_optimizer_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group(title="Optimizer")

    # Learning rate
    group.add_argument("--lr", type=float, default=1e-4,
                       help="Basic learning rate, varies depending on learning rate schedule and warmup.")
    group.add_argument("--warmup-min-lr", type=float, default=1e-6, help="Minimum learning rate for warmup.")
    group.add_argument("--warmup-num-steps", type=int, default=0, help="Number of warmup steps for learning rate.")

    # Optimizer
    group.add_argument("--adam-beta1", type=float, default=0.9,
                       help="[AdamW] First coefficient for computing running averages of gradient.")
    group.add_argument("--adam-beta2", type=float, default=0.999,
                       help="[AdamW] Second coefficient for computing running averages of gradient square.")
    group.add_argument("--adam-eps", type=float, default=1e-8,
                       help="[AdamW] Term added to the denominator to improve numerical stability.")
    group.add_argument("--weight-decay", type=float, default=0,
                       help="Weight decay coefficient for L2 regularization.")
    return parser

def add_train_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group(title="HunyuanVideo train args")


    return parser

def add_network_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group(title="HunyuanVideo network args")

    # Main model
    group.add_argument(
        "--model",
        type=str,
        choices=list(HUNYUAN_VIDEO_CONFIG.keys()),
        default="HYVideo-T/2-cfgdistill",
    )
    group.add_argument(
        "--latent-channels",
        type=str,
        default=16,
        help="Number of latent channels of DiT. If None, it will be determined by `vae`. If provided, "
        "it still needs to match the latent channels of the VAE model.",
    )
    group.add_argument(
        "--precision",
        type=str,
        default="bf16",
        choices=PRECISIONS,
        help="Precision mode. Options: fp32, fp16, bf16. Applied to the backbone model and optimizer.",
    )

    # RoPE
    group.add_argument(
        "--rope-theta", type=int, default=256, help="Theta used in RoPE."
    )

    group.add_argument("--gradient-checkpoint", action="store_true",
                       help="Enable gradient checkpointing to reduce memory usage.")

    group.add_argument("--gradient-checkpoint-layers", type=int, default=-1,
                       help="Number of layers to checkpoint. -1 for all layers. `n` for the first n layers.")

    return parser


def add_extra_models_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group(
        title="Extra models args, including vae, text encoders and tokenizers)"
    )

    # - VAE
    group.add_argument(
        "--vae",
        type=str,
        default="884-16c-hy",
        choices=list(VAE_PATH),
        help="Name of the VAE model.",
    )
    group.add_argument(
        "--vae-precision",
        type=str,
        default="fp16",
        choices=PRECISIONS,
        help="Precision mode for the VAE model.",
    )
    group.add_argument(
        "--vae-tiling",
        action="store_true",
        help="Enable tiling for the VAE model to save GPU memory.",
    )
    group.set_defaults(vae_tiling=True)

    group.add_argument(
        "--text-encoder",
        type=str,
        default="llm-i2v",
        choices=list(TEXT_ENCODER_PATH),
        help="Name of the text encoder model.",
    )
    group.add_argument(
        "--text-encoder-precision",
        type=str,
        default="fp16",
        choices=PRECISIONS,
        help="Precision mode for the text encoder model.",
    )
    group.add_argument(
        "--text-states-dim",
        type=int,
        default=4096,
        help="Dimension of the text encoder hidden states.",
    )
    group.add_argument(
        "--text-len", type=int, default=256, help="Maximum length of the text input."
    )
    group.add_argument(
        "--tokenizer",
        type=str,
        default="llm-i2v",
        choices=list(TOKENIZER_PATH),
        help="Name of the tokenizer model.",
    )
    group.add_argument(
        "--prompt-template",
        type=str,
        default="dit-llm-encode-i2v",
        choices=PROMPT_TEMPLATE,
        help="Image prompt template for the decoder-only text encoder model.",
    )
    group.add_argument(
        "--prompt-template-video",
        type=str,
        default="dit-llm-encode-video-i2v",
        choices=PROMPT_TEMPLATE,
        help="Video prompt template for the decoder-only text encoder model.",
    )
    group.add_argument(
        "--hidden-state-skip-layer",
        type=int,
        default=2,
        help="Skip layer for hidden states.",
    )
    group.add_argument(
        "--apply-final-norm",
        action="store_true",
        help="Apply final normalization to the used text encoder hidden states.",
    )

    # - CLIP
    group.add_argument(
        "--text-encoder-2",
        type=str,
        default="clipL",
        choices=list(TEXT_ENCODER_PATH),
        help="Name of the second text encoder model.",
    )
    group.add_argument(
        "--text-encoder-precision-2",
        type=str,
        default="fp16",
        choices=PRECISIONS,
        help="Precision mode for the second text encoder model.",
    )
    group.add_argument(
        "--text-states-dim-2",
        type=int,
        default=768,
        help="Dimension of the second text encoder hidden states.",
    )
    group.add_argument(
        "--tokenizer-2",
        type=str,
        default="clipL",
        choices=list(TOKENIZER_PATH),
        help="Name of the second tokenizer model.",
    )
    group.add_argument(
        "--text-len-2",
        type=int,
        default=77,
        help="Maximum length of the second text input.",
    )

    return parser


def add_denoise_schedule_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group(title="Denoise schedule args")

    group.add_argument(
        "--denoise-type",
        type=str,
        default="flow",
        help="Denoise type for noised inputs.",
    )

    # Flow Matching
    group.add_argument(
        "--flow-shift",
        type=float,
        default=17.0,
        help="Shift factor for flow matching schedulers.",
    )
    group.add_argument(
        "--flow-reverse",
        action="store_true",
        help="If reverse, learning/sampling from t=1 -> t=0.",
    )
    group.add_argument(
        "--flow-solver",
        type=str,
        default="euler",
        help="Solver for flow matching.",
    )
    group.add_argument(
        "--use-linear-quadratic-schedule",
        action="store_true",
        help="Use linear quadratic schedule for flow matching."
        "Following MovieGen (https://ai.meta.com/static-resource/movie-gen-research-paper)",
    )
    group.add_argument(
        "--linear-schedule-end",
        type=int,
        default=25,
        help="End step for linear quadratic schedule for flow matching.",
    )

    return parser


def add_inference_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group(title="Inference args")

    # ======================== Model loads ========================
    group.add_argument(
        "--model-base",
        type=str,
        default="ckpts",
        help="Root path of all the models, including t2v models and extra models.",
    )
    group.add_argument(
        "--dit-weight",
        type=str,
        default="ckpts/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt",
        help="Path to the HunyuanVideo model. If None, search the model in the args.model_root."
        "1. If it is a file, load the model directly."
        "2. If it is a directory, search the model in the directory. Support two types of models: "
        "1) named `pytorch_model_*.pt`"
        "2) named `*_model_states.pt`, where * can be `mp_rank_00`.",
    )
    group.add_argument(
        "--i2v-dit-weight",
        type=str,
        default="ckpts/hunyuan-video-i2v-720p/transformers/mp_rank_00_model_states.pt",
        help="Path to the HunyuanVideo model. If None, search the model in the args.model_root."
        "1. If it is a file, load the model directly."
        "2. If it is a directory, search the model in the directory. Support two types of models: "
        "1) named `pytorch_model_*.pt`"
        "2) named `*_model_states.pt`, where * can be `mp_rank_00`.",
    )
    group.add_argument(
        "--model-resolution",
        type=str,
        default="540p",
        choices=["540p", "720p"],
        help="Root path of all the models, including t2v models and extra models.",
    )
    group.add_argument(
        "--load-key",
        type=str,
        default="module",
        help="Key to load the model states. 'module' for the main model, 'ema' for the EMA model.",
    )
    group.add_argument(
        "--use-cpu-offload",
        action="store_true",
        help="Use CPU offload for the model load.",
    )

    # ======================== Inference general setting ========================
    group.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for inference and evaluation.",
    )
    # 20250329 pftq: vary up the CFG and steps on each video in the batch
    group.add_argument(
        "--variety-batch",
        action="store_true",
        help="Vary up the CFG and steps on each video in the batch.",
    )
    group.add_argument(
        "--infer-steps",
        type=int,
        default=50,
        help="Number of denoising steps for inference.",
    )
    group.add_argument(
        "--disable-autocast",
        action="store_true",
        help="Disable autocast for denoising loop and vae decoding in pipeline sampling.",
    )
    group.add_argument(
        "--save-path",
        type=str,
        default="./results",
        help="Path to save the generated samples.",
    )
    group.add_argument(
        "--save-path-suffix",
        type=str,
        default="",
        help="Suffix for the directory of saved samples.",
    )
    group.add_argument(
        "--name-suffix",
        type=str,
        default="",
        help="Suffix for the names of saved samples.",
    )
    group.add_argument(
        "--num-videos",
        type=int,
        default=1,
        help="Number of videos to generate for each prompt.",
    )
    # ---sample size---
    group.add_argument(
        "--video-size",
        type=int,
        nargs="+",
        default=(720, 1280),
        help="Video size for training. If a single value is provided, it will be used for both height "
        "and width. If two values are provided, they will be used for height and width "
        "respectively.",
    )
    group.add_argument(
        "--video-length",
        type=int,
        default=129,
        help="How many frames to sample from a video. if using 3d vae, the number should be 4n+1",
    )
    # --- prompt ---
    group.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Prompt for sampling during evaluation.",
    )
    group.add_argument(
        "--seed-type",
        type=str,
        default="auto",
        choices=["file", "random", "fixed", "auto"],
        help="Seed type for evaluation. If file, use the seed from the CSV file. If random, generate a "
        "random seed. If fixed, use the fixed seed given by `--seed`. If auto, `csv` will use the "
        "seed column if available, otherwise use the fixed `seed` value. `prompt` will use the "
        "fixed `seed` value.",
    )
    group.add_argument("--seed", type=int, default=None, help="Seed for evaluation.")

    # Classifier-Free Guidance
    group.add_argument(
        "--neg-prompt", type=str, default=None, help="Negative prompt for sampling."
    )
    group.add_argument(
        "--cfg-scale", type=float, default=1.0, help="Classifier free guidance scale."
    )
    group.add_argument(
        "--embedded-cfg-scale",
        type=float,
        default=None,
        help="Embeded classifier free guidance scale.",
    )

    group.add_argument(
        "--use-fp8",
        action="store_true",
        help="Enable use fp8 for inference acceleration."
    )

    group.add_argument(
        "--reproduce",
        action="store_true",
        help="Enable reproducibility by setting random seeds and deterministic algorithms.",
    )

    return parser

def add_i2v_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group(title="I2V args")

    group.add_argument(
        "--i2v-mode",
        action="store_true",
        help="Whether to open i2v mode."
    )

    group.add_argument(
        "--i2v-resolution",
        type=str,
        default="720p",
        choices=["720p", "540p", "360p"],
        help="Resolution for i2v inference."
    )

    group.add_argument(
        "--i2v-image-path",
        type=str,
        default="./assets/demo/i2v/imgs/0.png",
        help="Image path for i2v inference."
    )

    group.add_argument(
        "--i2v-condition-type",
        type=str,
        default="token_replace",
        choices=["token_replace", "latent_concat"],
        help="Condition type for i2v model."
    )

    group.add_argument(
        "--i2v-stability", action="store_true", help="Whether to use i2v stability mode."
    )

    return parser


def add_lora_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group(title="lora args")

    group.add_argument(
        "--use-lora", action="store_true", help="Whether to open lora mode."
    )

    group.add_argument(
        "--lora-path", type=str, default="", help="Weight path for lora model."
    )

    group.add_argument(
        "--lora-scale", type=float, default=1.0, help="Fusion scale for lora model."
    )

    group.add_argument(
        "--lora-rank", type=int, default=64, help="Rank for lora model."
    )

    return parser

def add_parallel_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group(title="Parallel args")

    # ======================== Model loads ========================
    group.add_argument(
        "--ulysses-degree",
        type=int,
        default=1,
        help="Ulysses degree for xdit parallel args.",
    )
    group.add_argument(
        "--ring-degree",
        type=int,
        default=1,
        help="Ring degree for xdit parallel args.",
    )

    return parser


def sanity_check_args(args):
    # VAE channels
    vae_pattern = r"\d{2,3}-\d{1,2}c-\w+"
    if not re.match(vae_pattern, args.vae):
        raise ValueError(
            f"Invalid VAE model: {args.vae}. Must be in the format of '{vae_pattern}'."
        )
    vae_channels = int(args.vae.split("-")[1][:-1])
    if args.latent_channels is None:
        args.latent_channels = vae_channels
    if vae_channels != args.latent_channels:
        raise ValueError(
            f"Latent channels ({args.latent_channels}) must match the VAE channels ({vae_channels})."
        )
    return args
