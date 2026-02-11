import torch
from einops import rearrange

def ssitrim_loss(pred_depth, target_depth, mask=None, ignore_large_loss=0.0, smooth_transition=True, range_utilization=0.0, eps=1e-8, inverse_depth=True):
    """
    pred_depth shape: [B, V, 1, H, W]
    target_depth shape: [B, V, 1, H, W]
    impl according to MiDaS paper Eq.6 and Splatter-a-Video paper Eq.4
    Args:
        pred_depth: predicted depth
        target_depth: ground truth depth
        mask: validity mask
        ignore_large_loss: threshold for filtering large losses
        smooth_transition: whether to use smooth transition between small and large losses
        range_utilization: weight for encouraging better utilization of depth range
        eps: small constant for numerical stability
    """
    # (B, H * W)
    if mask is None:
        mask = torch.ones_like(target_depth)

    if pred_depth.dim() == 5:
        # seperate depth
        # pred_depth = rearrange(pred_depth, 'B V C H W -> (B V) C H W')
        # target_depth = rearrange(target_depth, 'B V C H W -> (B V) C H W')
        # mask = rearrange(mask, 'B V C H W -> (B V) C H W')
        # unify depth across all views
        pred_depth = rearrange(pred_depth, 'B V C H W -> B (V C) H W')
        target_depth = rearrange(target_depth, 'B V C H W -> B (V C) H W')
        mask = rearrange(mask, 'B V C H W -> B (V C) H W')
    
    # Add range utilization encouragement
    if range_utilization > 0:
        # Calculate min and max depth values
        min_depth = pred_depth.min(dim=1, keepdim=True)[0]
        max_depth = pred_depth.max(dim=1, keepdim=True)[0]
        
        # Encourage min to be close to 0 and max to be close to 1
        min_penalty = torch.relu(min_depth)  # Penalize if min > 0
        max_penalty = torch.relu(1 - max_depth)  # Penalize if max < 1
        
        # Encourage spread by penalizing small ranges
        range_size = max_depth - min_depth
        range_penalty = torch.relu(0.5 - range_size)  # Penalize if range < 0.5
        
        # Combine penalties
        utilization_loss = (min_penalty + max_penalty + range_penalty).mean() * range_utilization
    else:
        utilization_loss = 0.0
    
    if inverse_depth:
        # Convert to inverse depth
        pred_depth = 1.0 / (pred_depth + eps)
    
    pred_depth, target_depth = pred_depth * mask.float(), target_depth * mask.float()

    pred_depth, target_depth = pred_depth.flatten(1), target_depth.flatten(1)

    pix_num = pred_depth.shape[1]
    
    pred_t = torch.median(pred_depth.float(), dim=1).values
    target_t = torch.median(target_depth.float(), dim=1).values

    pred_s = (pred_depth - pred_t[:, None]).abs().sum(1) / pix_num + eps
    target_s = (target_depth - target_t[:, None]).abs().sum(1) / pix_num + eps

    pred_depth = (pred_depth - pred_t[:, None]) / pred_s[:, None]

    target_depth = (target_depth - target_t[:, None]) / target_s[:, None]

    delta = (pred_depth - target_depth)
    if ignore_large_loss > 0:
        delta_sq = delta ** 2
        if smooth_transition:
            # Smooth transition using sigmoid
            transition_width = ignore_large_loss * 0.2  # 20% of threshold for transition
            weight = torch.sigmoid(-(delta_sq - ignore_large_loss) / transition_width)
            # Clip gradients for stability
            with torch.no_grad():
                grad_scale = torch.clamp(weight, min=0.1)
            delta = delta * grad_scale
            loss_val = delta.abs().mean()
        else:
            # Original split logic with improved decay
            valid_mask = delta_sq < ignore_large_loss
            delta_small = delta[valid_mask]
            delta_large = delta[~valid_mask]
            # Improved decay weight with tunable rate
            decay_weight = torch.exp(-2 * (delta_large.abs().detach() - ignore_large_loss))
            # Clip gradients for stability
            decay_weight = torch.clamp(decay_weight, min=0.1)
            loss_val = delta_small.abs().mean() + (decay_weight * delta_large).abs().mean()
    else:
        loss_val = delta.abs().mean()

    # Add range utilization encouragement to the final loss
    return loss_val + utilization_loss

def ssimse_loss(pred_depth, target_depth, mask=None, ignore_large_loss=0.0, eps=1e-8):
    """
    Scale and translation invariant MSE loss
    Args:
        pred_depth: predicted depth [B, 1, H, W]
        target_depth: ground truth depth [B, 1, H, W]
        mask: validity mask [B, 1, H, W]
        ignore_large_loss: threshold for filtering large losses
        eps: small constant for numerical stability
    """
    if mask is None:
        mask = torch.ones_like(target_depth)
    
    # Apply mask and flatten
    mask = mask.float()
    pred_depth = pred_depth * mask
    target_depth = target_depth * mask
    
    pred_depth = pred_depth.flatten(1)  # [B, H*W]
    target_depth = target_depth.flatten(1)
    mask_flat = mask.flatten(1)
    
    # Compute means on valid pixels only
    valid_pixels = mask_flat.sum(1) + eps
    gt_mean = (target_depth * mask_flat).sum(1) / valid_pixels
    pred_mean = (pred_depth * mask_flat).sum(1) / valid_pixels
    
    # Center the depths
    pred_centered = pred_depth - pred_mean[:, None] 
    target_centered = target_depth - gt_mean[:, None]
    
    # Compute scale factor (least squares)
    numerator = (pred_centered * target_centered * mask_flat).sum(1)
    denominator = (target_centered**2 * mask_flat).sum(1) + eps
    
    # Clamp scale factor for stability
    s = (numerator / denominator).clamp(-10, 10)
    t = pred_mean - s * gt_mean
    
    # Apply scale and translation
    pred_aligned = s[:, None] * pred_depth + t[:, None]
    
    # Compute loss on valid pixels only
    delta = (pred_aligned - target_depth) * mask_flat
    
    if ignore_large_loss > 0:
        valid_mask = ((delta ** 2) < ignore_large_loss) & (mask_flat > 0)
        if valid_mask.any():
            delta = delta[valid_mask]
        else:
            return torch.tensor(0.0, device=pred_depth.device)
    
    loss_val = (delta ** 2).mean()
    return loss_val