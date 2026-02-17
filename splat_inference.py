import torch
import tyro
import os
import numpy as np
import cv2
import kiui
from safetensors.torch import load_file
from collections import OrderedDict
import torch.nn.functional as F
import argparse
import random
from copy import deepcopy
from model.splat_model_inference import SplatModel
from configs.options_inference import AllConfigs
import imageio


def preprocess_data(frames, depths, timestamps, device):
    frames = torch.from_numpy(np.stack(frames)).float().to(device) / 255.0 # [V, H, W, C] -> [V, C, H, W]
    frames = frames.permute(0, 3, 1, 2).unsqueeze(0) # [1, V, C, H, W]

    depths = torch.from_numpy(np.stack(depths)).float().to(device) # [V, H, W]
    depths = depths.unsqueeze(1).unsqueeze(0) # [1, V, 1, H, W]

    timestamps = torch.tensor(timestamps, dtype=torch.float32, device=device).unsqueeze(0) # [1, V]

    timestamps = timestamps / (timestamps[..., -1].unsqueeze(-1))

    max_depth = depths.flatten(1).max(dim=1)[0][:, None, None, None, None]
    min_depth = depths.flatten(1).min(dim=1)[0][:, None, None, None, None]
    input_depths = (depths - min_depth) / (max_depth - min_depth + 1e-8)

    return frames, input_depths, timestamps

def get_image(path, H, W):
    """Load and resize an image."""
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)
    return img

def get_depth(path, H, W):
    """Load and resize a depth map."""
    if path.endswith('.exr'):
        depth = cv2.imread(path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_UNCHANGED)
    else:
        depth = cv2.imread(path, cv2.IMREAD_ANYDEPTH)
    if depth is None:
        depth = np.zeros((H, W), dtype=np.float32)
    else:
        depth = cv2.resize(depth.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
    return depth

def load_model_weights(model, decoder_path, device, compile=False):
    """Loads weights from the trained decoder checkpoint."""
    
    if decoder_path and os.path.exists(decoder_path):
        print(f"Loading all weights from {decoder_path}")
        state_dict_dec = load_file(decoder_path, device=device)
        new_state_dict_dec = OrderedDict()
        for k, v in state_dict_dec.items():
            if "_orig_mod." in k and not compile:
                k = k.replace('_orig_mod.', '')
            new_state_dict_dec[k] = v
        model.load_state_dict(new_state_dict_dec, strict=False)
    else:
        print(f"Decoder checkpoint path not found or not provided: {decoder_path}")

def main(opt: AllConfigs, args: argparse.Namespace):
    device = 'cuda'
    print(f"Using device: {device}")
    GAP = args.frame_gap

    torch.set_float32_matmul_precision('high')

    model_opt = deepcopy(opt)
    model_opt.input_frames = 1 
    model_opt.output_frames = GAP + 1
    model_opt.epoch = 0 
    model = SplatModel(model_opt).to(device)
    load_model_weights(model, opt.resume, device, opt.compile)
    model.eval()

    frames_dir = args.input_frames_path
    depths_dir = args.input_depths_path
    
    if not os.path.isdir(frames_dir):
        print(f"Error: Input frames directory not found: {frames_dir}")
        return
    
    if depths_dir and not os.path.isdir(depths_dir):
        print(f"Warning: Input depths directory not found: {depths_dir}")
        depths_dir = None

    output_dir = args.output_dir if args.output_dir else os.path.join(opt.workspace, "inference_output", os.path.basename(frames_dir))
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    image_extensions = ('.png', '.jpg', '.jpeg')
    all_frame_files = sorted([
        os.path.join(frames_dir, f) for f in os.listdir(frames_dir) 
        if f.lower().endswith(image_extensions)
    ])
    
    depth_extensions = ('.png', '.exr', '.npy')
    all_depth_files = []
    if depths_dir:
        all_depth_files = sorted([
            os.path.join(depths_dir, f) for f in os.listdir(depths_dir) 
            if f.lower().endswith(depth_extensions)
        ])

    if not all_frame_files:
        print(f"Error: No frame files found in {frames_dir}")
        return
    
    if opt.enable_depth and not all_depth_files:
        print(f"Warning: No depth files found, using zero depths.")

    total_available_frames = len(all_frame_files)
    frame_gap = args.frame_gap
    num_output_frames = frame_gap + 1

    selected_indices = list(range(0, total_available_frames, frame_gap))

    if len(selected_indices) < 2:
        print(f"Error: Not enough frames in the sequence for the specified gap.")
        return

    selected_frame_files = [all_frame_files[i] for i in selected_indices]
    selected_depth_files = [all_depth_files[i] for i in selected_indices if i < len(all_depth_files)] if all_depth_files else []
    
    print(f"Loading {len(selected_frame_files)} frames with gap {frame_gap}: indices {selected_indices}")

    frames_data = [get_image(f, H=opt.image_height, W=opt.image_width) for f in selected_frame_files]
    
    if selected_depth_files and len(selected_depth_files) == len(selected_frame_files):
        depths_data = [get_depth(f, H=opt.image_height, W=opt.image_width) for f in selected_depth_files]
    else:
        print("Using zero depths for all frames.")
        depths_data = [np.zeros((opt.image_height, opt.image_width), dtype=np.float32) for _ in frames_data]

    selected_timestamps = np.array(list(range(len(selected_frame_files))), dtype=np.float32)
    frames, input_depths, timestamps = preprocess_data(frames_data, depths_data, selected_timestamps, device)
    
    original_frames = frames.clone()  # [1, V, C, H, W]
    
    V = frames.shape[1]
    B_new = V - 1  # Number of pairs

    if B_new <= 0:
        print(f"Error: Not enough frames ({V}) to form pairs.")
        return

    new_frames = torch.zeros((B_new, num_output_frames, frames.shape[2], frames.shape[3], frames.shape[4]), device=device, dtype=frames.dtype)
    new_depths = torch.zeros((B_new, num_output_frames, 1, input_depths.shape[3], input_depths.shape[4]), device=device, dtype=input_depths.dtype)
    new_timestamps = torch.zeros((B_new, num_output_frames), device=device, dtype=timestamps.dtype)
    
    fixed_timestamps = torch.linspace(0.0, 1.0, num_output_frames, device=device, dtype=timestamps.dtype)

    for k in range(B_new):
        frame_k = frames[0, k]
        depth_k = input_depths[0, k]
        frame_k_plus_1 = frames[0, k + 1]
        depth_k_plus_1 = input_depths[0, k + 1]

        new_frames[k, 0] = frame_k
        if num_output_frames > 2:
            new_frames[k, 1:-1] = frame_k.unsqueeze(0).repeat(num_output_frames - 2, 1, 1, 1)
        new_frames[k, -1] = frame_k_plus_1

        new_depths[k, 0] = depth_k
        if num_output_frames > 2:
            new_depths[k, 1:-1] = depth_k.unsqueeze(0).repeat(num_output_frames - 2, 1, 1, 1)
        new_depths[k, -1] = depth_k_plus_1

        new_timestamps[k] = fixed_timestamps

    data = {
        'frames': new_frames,
        'depths': new_depths,
        'timestamps': new_timestamps,
    }

    print(f"Rearranged data from [1, {V}, ...] to [{B_new}, {num_output_frames}, ...]")
    output_prefix = ""

    B_new = data['frames'].shape[0]

    # --- Inference ---
    print("Running inference...")
    with torch.no_grad():
        results = {}
        for i in range(0, B_new, args.batch_size):
            end_index = min(i + args.batch_size, B_new)
            if i + args.batch_size > B_new and opt.compile:
                print(f"Warning: Dropping last {B_new - i} samples to avoid batch size mismatch with torch.compile. Consider setting batch_size to a divisor of {B_new} or disable torch.compile.")
                break
            print(f"Processing frames {i+1} to {end_index}")
            pair_data = {
                'frames': data['frames'][i:end_index],
                'depths': data['depths'][i:end_index],
                'timestamps': data['timestamps'][i:end_index],
            }
            output = model(pair_data)
            results['pred_frames'] = results.get('pred_frames', []) + [output['pred_frames']]
        
        results['pred_frames'] = torch.cat(results['pred_frames'], dim=0) # [B_new, num_new_frames, C, H, W]
    
    i = 0
    while os.path.exists(os.path.join(output_dir, f"{output_prefix}render_video_{i}.mp4")):
        i += 1
    output_video_path = os.path.join(output_dir, f"{output_prefix}render_video_{i}.mp4")
    original_video_path = os.path.join(output_dir, f"{output_prefix}input_video_{i}.mp4")
    print(f"Saving videos to {output_dir} with index {i}...")
    video_writer = None
    
    if 'original_frames' in locals() and original_frames is not None:
        original_video_writer = None
        print(f"Saving original input video to {original_video_path}...")
        for frame_idx in range(original_frames.shape[1]):
            orig_frame_np = (original_frames[0, frame_idx].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            orig_frame_bgr = cv2.cvtColor(orig_frame_np, cv2.COLOR_RGB2BGR)

            if original_video_writer is None:
                original_video_writer = imageio.get_writer(original_video_path, fps=args.fps // GAP, codec='libx264', quality=8)
            original_video_writer.append_data(cv2.cvtColor(orig_frame_bgr, cv2.COLOR_BGR2RGB))

        if original_video_writer:
            original_video_writer.close()
            print(f"Original video saved to {original_video_path}")

    current_index = 0
    for pair_idx in range(results['pred_frames'].shape[0]):
        pred_frames_seq = results['pred_frames'][pair_idx]

        for frame_in_seq_idx in range(pred_frames_seq.shape[0]):
            if pair_idx > 0 and frame_in_seq_idx == 0:
                continue

            pred_frame_np = (pred_frames_seq[frame_in_seq_idx].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            pred_frame_bgr = cv2.cvtColor(pred_frame_np, cv2.COLOR_RGB2BGR)

            if video_writer is None:
                video_writer = imageio.get_writer(output_video_path, fps=args.fps, codec='libx264', quality=8)
            
            video_writer.append_data(cv2.cvtColor(pred_frame_bgr, cv2.COLOR_BGR2RGB))

            current_index += 1

    if video_writer:
        video_writer.close()
        print(f"Output video saved to {output_video_path}")
    
    print("Inference complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--input_frames_path", type=str, default=None, help="Path to input frames directory.", required=True)
    parser.add_argument("--input_depths_path", type=str, default=None, help="Path to input depths directory.", required=True)
    parser.add_argument("--output_dir", type=str, default="workspace_inference", help="Directory to save output videos.")
    parser.add_argument("--frame_gap", type=int, default=3, help="Gap between loaded frames.")
    parser.add_argument("--fps", type=int, default=24, help="Saved video fps.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for inference.")

    args, unknown_args = parser.parse_known_args()

    opt = tyro.cli(AllConfigs, args=unknown_args)

    main(opt, args)
