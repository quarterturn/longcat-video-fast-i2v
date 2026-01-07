import os
import sys
import time
import datetime
import numpy as np

import torch
import torch.distributed as dist

torch.set_float32_matmul_precision("high")  # Faster matmuls with negligible quality loss

from transformers import AutoTokenizer, UMT5EncoderModel
from transformers import BitsAndBytesConfig as TransformersBitsAndBytesConfig
from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig
from PIL import Image
from torchvision.io import write_video

sys.path.append("./LongCat-Video")
from longcat_video.pipeline_longcat_video import LongCatVideoPipeline
from longcat_video.modules.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)
from longcat_video.modules.autoencoder_kl_wan import AutoencoderKLWan
from longcat_video.modules.longcat_video_dit import (
    LongCatVideoTransformer3DModel,
)
from longcat_video.context_parallel import context_parallel_util
from longcat_video.context_parallel.context_parallel_util import (
    init_context_parallel,
)

from utils import get_args, strify, GiB
import cache_dit


def torch_gc():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


def generate(args):
    print(args)

    # User inputs
    image_path = args.image_path
    prompt = args.prompt or ""
    negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
    num_frames = 81 if args.frames is None else args.frames
    num_segments = args.num_segments
    num_cond_frames = args.num_cond_frames
    use_distill_first = args.use_distill_first

    if image_path is None:
        raise ValueError("--image_path is required for long image-to-video generation")

    # Load and resize input image
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = Image.open(image_path).convert("RGB")
    original_width, original_height = image.size
    if original_height != 480:
        scale_factor = 480 / original_height
        new_height = 480
        new_width = int(original_width * scale_factor + 0.5)
        new_width = (new_width // 16) * 16
        image = image.resize((new_width, new_height), Image.BICUBIC)
    target_size = image.size

    checkpoint_dir = args.checkpoint_dir
    context_parallel_size = args.context_parallel_size

    # Distributed setup
    rank = int(os.environ["RANK"])
    num_gpus = torch.cuda.device_count()
    local_rank = rank % num_gpus
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=3600 * 24))

    global_rank = dist.get_rank()
    num_processes = dist.get_world_size()

    if context_parallel_size is None:
        context_parallel_size = num_processes

    init_context_parallel(
        context_parallel_size=context_parallel_size,
        global_rank=global_rank,
        world_size=num_processes,
    )
    cp_size = context_parallel_util.get_cp_size()
    cp_split_hw = context_parallel_util.get_optimal_split(cp_size)

    # Load components
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint_dir, subfolder="tokenizer", torch_dtype=torch.bfloat16
    )

    text_encoder = UMT5EncoderModel.from_pretrained(
        checkpoint_dir,
        subfolder="text_encoder",
        torch_dtype=torch.bfloat16,
        quantization_config=(
            TransformersBitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            # if args.quantize
            # else None
        ),
    )

    vae = AutoencoderKLWan.from_pretrained(
        checkpoint_dir, subfolder="vae", torch_dtype=torch.bfloat16
    )
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        checkpoint_dir, subfolder="scheduler", torch_dtype=torch.bfloat16
    )

    # Load DiT
    if args.quantize:
        if context_parallel_size >= 2 and GiB() >= 40:
            dit = LongCatVideoTransformer3DModel.from_pretrained(
                checkpoint_dir,
                subfolder="dit",
                cp_split_hw=cp_split_hw,
                torch_dtype=torch.bfloat16,
            )
            dit = cache_dit.quantize(dit, quant_type="fp8_w8a16_wo")
        else:
            dit = LongCatVideoTransformer3DModel.from_pretrained(
                checkpoint_dir,
                subfolder="dit",
                cp_split_hw=cp_split_hw,
                torch_dtype=torch.bfloat16,
                quantization_config=DiffusersBitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                ),
            )
    else:
        dit = LongCatVideoTransformer3DModel.from_pretrained(
            checkpoint_dir,
            subfolder="dit",
            cp_split_hw=cp_split_hw,
            torch_dtype=torch.bfloat16,
        )

    pipe = LongCatVideoPipeline(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        vae=vae,
        scheduler=scheduler,
        dit=dit,
    )

    if GiB() <= 48:
        pipe.vae.enable_tiling()
        pipe.vae.to(f"cuda:{local_rank}")  # Always safe for VAE

    # Only move DiT in full precision (fixes embedding device error)
    if not args.quantize:
        pipe.dit.to(f"cuda:{local_rank}")
        pipe.text_encoder.to(f"cuda:{local_rank}")  # This fixes the embedding device error

    # Cache-DiT
    if args.cache:
        from cache_dit import BlockAdapter, ForwardPattern, DBCacheConfig, TaylorSeerCalibratorConfig

        cache_dit.enable_cache(
            BlockAdapter(
                transformer=pipe.dit,
                blocks=pipe.dit.blocks,
                forward_pattern=ForwardPattern.Pattern_3,
                check_forward_pattern=False,
                has_separate_cfg=False,
            ),
            cache_config=DBCacheConfig(
                Fn_compute_blocks=args.Fn,
                Bn_compute_blocks=args.Bn,
                max_warmup_steps=args.max_warmup_steps,
                max_cached_steps=args.max_cached_steps,
                max_continuous_cached_steps=args.max_continuous_cached_steps,
                residual_diff_threshold=args.rdt,
                num_inference_steps=50 if args.steps is None else args.steps,
            ),
            calibrator_config=(
                TaylorSeerCalibratorConfig(taylorseer_order=args.taylorseer_order)
                if args.taylorseer
                else None
            ),
        )

    # Compile
    if args.compile:
        pipe.dit = torch.compile(pipe.dit)

        # Warmup
        _ = pipe.generate_vc(
            video=[image],
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_frames=num_frames,
            num_cond_frames=1,
            num_inference_steps=4,
            guidance_scale=4.0,
            generator=torch.Generator(device=local_rank).manual_seed(42 + global_rank),
        )
        torch_gc()

    generator = torch.Generator(device=local_rank).manual_seed(42 + global_rank)

    # Distill LoRA
    if use_distill_first:
        cfg_step_lora_path = os.path.join(checkpoint_dir, "lora/cfg_step_lora.safetensors")
        if os.path.exists(cfg_step_lora_path):
            pipe.dit.load_lora(cfg_step_lora_path, "cfg_step_lora")
            pipe.dit.enable_loras(["cfg_step_lora"])
            first_steps = 16
            first_guidance = 1.0
        else:
            first_steps = 50 if args.steps is None else args.steps
            first_guidance = 4.0
    else:
        first_steps = 50 if args.steps is None else args.steps
        first_guidance = 4.0

    start = time.time()

    # Initial segment
    output = pipe.generate_vc(
        video=[image],
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_frames=num_frames,
        num_cond_frames=1,
        num_inference_steps=first_steps,
        guidance_scale=first_guidance,
        generator=generator,
    )[0]

    if use_distill_first:
        pipe.dit.disable_all_loras()
        torch_gc()

    # Save initial
    if local_rank == 0:
        frames = [Image.fromarray((output[i] * 255).astype(np.uint8).clip(0, 255)) for i in range(output.shape[0])]
        frames = [f.resize(target_size, Image.BICUBIC) for f in frames]
        tensor = torch.from_numpy(np.array(frames))
        write_video("output_long_i2v_0.mp4", tensor, fps=15, video_codec="libx264", options={"crf": "18"})

    video = [Image.fromarray((output[i] * 255).astype(np.uint8).clip(0, 255)) for i in range(output.shape[0])]
    video = [f.resize(target_size, Image.BICUBIC) for f in video]
    del output
    torch_gc()

    all_generated_frames = video.copy()
    current_video = video

    # Extensions
    for segment_idx in range(num_segments):
        if local_rank == 0:
            print(f"Generating extension segment {segment_idx + 1}/{num_segments}...")

        output = pipe.generate_vc(
            video=current_video,
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_frames=num_frames,
            num_cond_frames=num_cond_frames,
            num_inference_steps=50 if args.steps is None else args.steps,
            guidance_scale=4.0,
            generator=generator,
        )[0]

        new_video = [Image.fromarray((output[i] * 255).astype(np.uint8).clip(0, 255)) for i in range(output.shape[0])]
        new_video = [f.resize(target_size, Image.BICUBIC) for f in new_video]
        del output
        torch_gc()

        all_generated_frames.extend(new_video[num_cond_frames:])
        current_video = new_video

        if local_rank == 0:
            tensor = torch.from_numpy(np.array(all_generated_frames))
            write_video(f"output_long_i2v_{segment_idx + 1}.mp4", tensor, fps=15, video_codec="libx264", options={"crf": "18"})
            del tensor
            torch_gc()

    if local_rank == 0:
        time_cost = time.time() - start
        print(f"Total generation time: {time_cost:.2f}s")
        final_tensor = torch.from_numpy(np.array(all_generated_frames))
        write_video("output_long_i2v_final.mp4", final_tensor, fps=15, video_codec="libx264", options={"crf": "18"})
        if args.cache:
            cache_dit.summary(pipe.dit)

    del all_generated_frames
    torch_gc()

    if dist.is_initialized():
        dist.destroy_process_group()


def _parse_args():
    DEFAULT_CHECKPOINT_DIR = os.environ.get("LONGCAT_VIDEO_DIR", None)
    parser = get_args(parse=False)
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--num_segments", type=int, default=11)
    parser.add_argument("--num_cond_frames", type=int, default=13, choices=[5, 9, 13])
    parser.add_argument("--use_distill_first", action="store_true")
    parser.add_argument("--frames", type=int, default=None)
    parser.add_argument("--context_parallel_size", type=int, default=None)
    parser.add_argument("--checkpoint_dir", type=str, default=DEFAULT_CHECKPOINT_DIR)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = _parse_args()
    generate(args)
