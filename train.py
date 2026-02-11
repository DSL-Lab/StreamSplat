import tyro
from safetensors.torch import load_file
from collections import OrderedDict
import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from datasets.provider_co3d import Co3DDataset as Co3DDataset
from datasets.provider_re10k_map import Re10kMapDataset as Re10kDataset
from datasets.provider_davis import DAVISDataset as DavisDataset
from datasets.provider_vos import VOSDataset as VOSDataset
from datasets.provider_combined import CombinedDataset
from datasets.augmentv2 import augment_batch
from model.encoder_model import StaticEncoder
from configs.options import AllConfigs
import wandb
import os
import datetime
import kiui
from utils.general_utils import CosineWarmupScheduler
import cv2
import warnings
import time


def main():
    set_seed(42)
    os.environ["WANDB__SERVICE_WAIT"] = "300"
    
    opt = tyro.cli(AllConfigs)

    torch.set_float32_matmul_precision('high')
    
    accelerator = Accelerator(
        mixed_precision=opt.mixed_precision,
        gradient_accumulation_steps=opt.gradient_accumulation_steps,
    )

    dataset_map = {
        "co3d": Co3DDataset,
        "re10k": Re10kDataset, 
        "davis": DavisDataset,
        'vos': VOSDataset,
        'combined': CombinedDataset
    }
    
    for key, Dataset in dataset_map.items():
        if key in opt.root_path.lower():
            dataset_nm = key
            print(f"Loading dataset: {key}")
            break
    else:
        raise ValueError(f"Dataset {opt.root_path} not supported")

    # Warmup parameters
    # initial_batch_size = 8
    initial_batch_size = opt.batch_size
    target_batch_size = opt.batch_size
    warmup_epochs = 15
    current_batch_size = initial_batch_size

    train_dataset = Dataset(opt=opt, shuffle=True, training=True)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size = current_batch_size,
        num_workers=opt.num_workers,
        pin_memory=True,
        shuffle=not isinstance(train_dataset, torch.utils.data.IterableDataset),
        drop_last=True,
    )

    test_dataset = Dataset(opt=opt, shuffle=True, training=False)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=opt.batch_size * 2,
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    
    model = StaticEncoder(opt)

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.lr, weight_decay=0.05, betas=(0.9, 0.95))
    
    if isinstance(train_dataset, torch.utils.data.IterableDataset):
        trainloader_len_all = sum(1 for _ in train_dataloader)
    else:
        trainloader_len_all = len(train_dataloader)
    
    steps_per_epoch = int(trainloader_len_all / opt.gradient_accumulation_steps)
    total_steps = opt.num_epochs * steps_per_epoch
    lr_decay_steps = opt.lr_decay_epochs * steps_per_epoch
    warmup_iters = opt.warmup_iters

    scheduler = CosineWarmupScheduler(optimizer=optimizer, warmup_iters=warmup_iters, max_iters=lr_decay_steps, min_lr=0.1*opt.lr, decay=True) 

    if opt.resume is not None:
        if os.path.exists(opt.resume):
            print(f"Loading resume file from {opt.resume}")
            state_dict = load_file(opt.resume)
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if "_orig_mod" in k and not opt.compile:
                    k = k.replace('_orig_mod.', '')
                
                if k in model.state_dict() and model.state_dict()[k].shape == v.shape:
                    if opt.mixed_precision == 'bf16':
                        new_state_dict[k] = v
                    else:
                        new_state_dict[k] = v.type(torch.float32)
                else:
                    if k not in model.state_dict():
                        print(f"Key {k} not in model state dict")
                    else:
                        print(f"Skipping {k} due to shape mismatch: {v.shape} vs {model.state_dict().get(k, 'N/A').shape}")
            
            load_result = model.load_state_dict(new_state_dict, strict=False)
            print("Loaded resume file")
            if accelerator.is_main_process:
                print("Missing keys:", len(load_result.missing_keys))
                print("Unexpected keys:", len(load_result.unexpected_keys))
        else:
            print(f"Resume file {opt.resume} not found")

    if accelerator.is_main_process:
        print(f"steps_per_epoch: {steps_per_epoch}, total_steps: {total_steps}")
        print(f"opt: {opt}")

    # Prepare with accelerator
    model, optimizer, train_dataloader, test_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, test_dataloader, scheduler
    )

    # Load checkpoint if exists
    state_files = ['optimizer.bin', 'scheduler.bin', 'model.safetensors']
    start_epoch = 0
    base_dir = os.path.join(opt.workspace, 'checkpoint_latest')
    state_exists = all(os.path.exists(os.path.join(base_dir, f)) for f in state_files)
    try:
        if state_exists:
            accelerator.load_state(base_dir, strict=False)
            start_epoch = int(scheduler.scheduler._epoch)
            print(f"Resuming from {base_dir} at epoch {start_epoch}")   
            current_batch_size = min(initial_batch_size * (2 ** (start_epoch // warmup_epochs)), target_batch_size)
            train_dataloader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=current_batch_size,
                num_workers=opt.num_workers,
                pin_memory=True,
                shuffle=not isinstance(train_dataset, torch.utils.data.IterableDataset),
                drop_last=True,
            )
            train_dataloader = accelerator.prepare(train_dataloader)
            print(f"Updated current batch size to {current_batch_size}")
        else:
            print(f"Starting from scratch at epoch {start_epoch}")
    except Exception as e:
        print(f"Error loading state: {e}, starting from scratch at epoch {start_epoch}") 

    if isinstance(train_dataset, torch.utils.data.IterableDataset):
        # if dist train and iterable dataset, need to calculate len of per rank,
        trainloader_len_rank = sum(1 for _ in train_dataloader)
    else:
        # if not dist train or not iterable dataset, then len of all
        trainloader_len_rank = len(train_dataloader)
    
    if isinstance(test_dataset, torch.utils.data.IterableDataset):
        testloader_len_rank = sum(1 for _ in test_dataloader)
    else:
        testloader_len_rank = len(test_dataloader)
    
    if accelerator.is_main_process:
        wandb.init(
            project="streamsplat_{:s}".format(dataset_nm),
            config=opt,
            dir=opt.workspace,
            name="encoder",
        )
        wandb.watch(model, log_freq=500)

    accelerator.wait_for_everyone() 
    start_time = datetime.datetime.now()
    last_checkpoint_time = time.time()
    for epoch in range(start_epoch, opt.num_epochs):
        # double batch size every warmup_epochs until target_batch_size is reached
        if epoch > 0 and epoch % warmup_epochs == 0 and current_batch_size < target_batch_size:
            current_batch_size = min(current_batch_size * 2, target_batch_size)
            # Update DataLoader with new batch size
            train_dataloader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=current_batch_size,
                num_workers=opt.num_workers,
                pin_memory=True,
                shuffle=not isinstance(train_dataset, torch.utils.data.IterableDataset),
                drop_last=True,
            )
            train_dataloader = accelerator.prepare(train_dataloader)
            trainloader_len_rank = len(train_dataloader)
            if accelerator.is_main_process:
                print(f"Doubling batch size to {current_batch_size}")
                print(f"Length of train dataloader: {trainloader_len_rank}")
            accelerator.wait_for_everyone()
        model.train()
        total_loss = 0.0
        total_psnr = 0.0
        mse_loss = 0.0
        depth_loss = 0.0
        total_lpips = 0.0
        total_ssim = 0.0
        opt.epoch = epoch

        if accelerator.is_main_process:
            progress_bar = tqdm(
                train_dataloader, 
                desc=f"Epoch {epoch}/{opt.num_epochs}", 
                disable=not accelerator.is_main_process
            )
        else:
            progress_bar = train_dataloader

        if isinstance(train_dataset, torch.utils.data.IterableDataset):
            torch.manual_seed(epoch + 42)  # random shuffle data
        for i, data in enumerate(progress_bar):
            with accelerator.accumulate(model):

                step_ratio = (epoch + i / trainloader_len_rank) / opt.num_epochs

                if opt.use_augmentation:
                    data = augment_batch(data)
                
                out = model(data, step_ratio)
                loss = out['loss']
                psnr_value = out['psnr']
                lpips_value = out.get('loss_lpips', torch.tensor(0.0, device=loss.device))
                ssim_value = out.get('ssim', torch.tensor(0.0, device=loss.device))
                
                accelerator.backward(loss)

                for name, param in model.named_parameters():
                    if param.grad is not None:
                        if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                        torch.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)
                
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), opt.gradient_clip)
                
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step(epoch)
                
                total_loss += loss.detach()
                mse_loss += out['mse_loss'].detach()
                depth_loss += out['depth_loss'].detach()
                total_psnr += psnr_value.detach()
                total_lpips += lpips_value.detach()
                total_ssim += ssim_value.detach()

                if accelerator.is_main_process:
                    progress_bar.set_postfix(psnr=psnr_value.item(), lpips=lpips_value.item())

            # Save checkpoint every 30 minutes
            accelerator.wait_for_everyone()
            current_time = time.time()
            if current_time - last_checkpoint_time >= 1800:  # 30 minutes = 1800 seconds
                accelerator.wait_for_everyone()
                accelerator.save_state(output_dir=os.path.join(opt.workspace, 'checkpoint_latest'))
                last_checkpoint_time = current_time
                accelerator.wait_for_everyone()

        total_loss = accelerator.gather_for_metrics(total_loss).mean()
        total_mse_loss = accelerator.gather_for_metrics(mse_loss).mean()
        total_depth_loss = accelerator.gather_for_metrics(depth_loss).mean()
        total_psnr = accelerator.gather_for_metrics(total_psnr).mean()
        total_lpips = accelerator.gather_for_metrics(total_lpips).mean()
        total_ssim = accelerator.gather_for_metrics(total_ssim).mean()

        if accelerator.is_main_process:
            total_loss /= trainloader_len_rank
            total_psnr /= trainloader_len_rank
            total_mse_loss /= trainloader_len_rank
            total_depth_loss /= trainloader_len_rank
            total_lpips /= trainloader_len_rank
            total_ssim /= trainloader_len_rank

            mem_free, mem_total = torch.cuda.mem_get_info()  
            current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')  
            current_lr = scheduler.get_last_lr()[0]
            elapsed = datetime.datetime.now() - start_time
            elapsed_str = str(elapsed).split('.')[0]  
            print(f"[{current_time} INFO] {epoch}/{opt.num_epochs} | "
                f"Elapsed: {elapsed_str} | "
                f"Mem: {(mem_total-mem_free)/1024**3:.2f}/{mem_total/1024**3:.2f}G | "
                f"LR: {scheduler.get_last_lr()[0]:.7f} | "
                f"Loss: {total_loss.item():.6f} | PSNR: {total_psnr.item():.4f} | "
                f"LPIPS: {total_lpips.item():.4f} | SSIM: {total_ssim.item():.4f} | ")

            wandb.log({"Loss/train": total_loss, "Loss/mse": total_mse_loss, "Loss/depth": total_depth_loss, "Loss/reg": total_reg_loss,
                       "train/PSNR": total_psnr,
                       "train/LPIPS": total_lpips,
                       "train/SSIM": total_ssim,
                       "LR/lr": scheduler.get_last_lr()[0],
                      }, step=epoch, commit=True)
            
        if epoch % 10 == 0 or epoch == opt.num_epochs - 1:
            accelerator.wait_for_everyone()
            accelerator.save_state(output_dir=os.path.join(opt.workspace, 'checkpoint_ep{:03d}'.format(epoch)))
        accelerator.wait_for_everyone()
        
    print("\nTraining complete.")


if __name__ == "__main__":
    main()
