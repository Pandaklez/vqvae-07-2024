import random, os, sys

print(sys.executable)
os.environ['WANDB__EXECUTABLE'] = '/nfs/deepspeech/home/annkle/DWPose/miniconda3/envs/vqvae_anya/bin/python3'

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torchvision.utils import save_image, make_grid
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
import utils
import pandas as pd
import io
import time
import zipfile
import protobuf
import wandb
import torch.optim as optim
from torch.optim import AdamW, Adam
from sklearn.model_selection import train_test_split
from vector_quantize_pytorch import ResidualVQ

import utils
import importlib
utils = importlib.reload(utils)
import acrh_from_norm_meanpose_nowarmup as arch
arch = importlib.reload(arch)
import argparse


def load_data(path_to_data, output_dim=112, normalize_by_mean_pose=True, num_workers=0, batch_size=256, sequence_length=30):
    """ load up to torch.utils.data.DataLoader """
    sslc_pose = pd.read_csv('sslc_pose_2.csv', encoding='utf-8')  # ./SSL_video_eaf/sslc_pose_2.csv
    len_data = len(sslc_pose)

    if path_to_data.endswith(".zip"):
        dataset = utils.ZipPoseDataset(
            path_to_data, in_memory=True,
            dtype=torch.float32,
            max_length=output_dim,
            df=sslc_pose
        )
        threshold_10 = round(len_data * 0.1)
        training_dataset = dataset.slice(threshold_10, None)  # TODO: change to 10%
        test_dataset = dataset.slice(0, threshold_10)   # TODO: change to 10%
        shuffle = True  # Shuffle is only slow without in_memory since the zip file is read sequentially
        num_workers = 0  # Reading from multiple workers errors out since the zip file is read sequentially
        # training_iter_dataset = utils.PackedDataset(training_dataset, max_length=sequence_length, shuffle=shuffle)
        train_loader = DataLoader(
            training_dataset, batch_size=batch_size,
            num_workers=num_workers
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size,
            shuffle=False, num_workers=num_workers
        )
    else:
        train_df, test_df = train_test_split(sslc_pose, test_size=0.2, random_state=42)
        training_dataset = utils.PoseDataset(
            df=train_df, root_dir='./SSL_video_eaf/SSLC_poses/',
            sequence_length=sequence_length, normalize_by_mean_pose=normalize_by_mean_pose
        )
        test_dataset = utils.PoseDataset(
            df=test_df, root_dir='./SSL_video_eaf/SSLC_poses/',   # ./SSL_video_eaf/SSLC_poses/
            sequence_length=sequence_length, normalize_by_mean_pose=normalize_by_mean_pose
        )
        kwargs = {'num_workers': num_workers, 'pin_memory': True} 
        train_loader = DataLoader(dataset=training_dataset, batch_size=batch_size, shuffle=False, **kwargs)
        test_loader  = DataLoader(dataset=test_dataset,  batch_size=batch_size, shuffle=False,  **kwargs)
    return training_dataset, test_dataset, train_loader, test_loader


def train(model, optimizer, mse_loss, train_loader, epochs, print_step, device, wandb, codebook, sequence_length, use_amp=False):
    print("Start training VQ-VAE...")
    model.train()

    # Constructs a ``scaler`` once, at the beginning of the convergence run, using default arguments.
    # If your network fails to converge with default ``GradScaler`` arguments, please file an issue.
    # The same ``GradScaler`` instance should be used for the entire convergence run.
    # If you perform multiple convergence runs in the same script, each run should use
    # a dedicated fresh ``GradScaler`` instance. ``GradScaler`` instances are lightweight.
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    for epoch in range(epochs):
        commitment_beta = arch.calculate_beta_log(epoch+1, total_iterations=epochs+1, initial_beta=0.35, final_beta=0.001, smoothing_factor=0.1)
        for batch_idx, x in enumerate(train_loader):
            
            # optimizer.zero_grad()
            
            if x is None:
                ("x is None")
                continue
                
                # https://pytorch.org/docs/stable/amp.html#autocast-op-reference
            """
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
                x = x.to(device, torch.float16)
                print("x.device: ", x.device)

                x_hat, commitment_loss, codebook_loss, perplexity = model(x, epoch)
                print("x_hat.device: ", x_hat.device)
                assert x_hat.dtype is torch.float16

                print("commitment_loss.device, codebook_loss.device: ", commitment_loss.device, codebook_loss.device)

                recon_loss = mse_loss(x_hat, x)
                print("recon_loss: ", recon_loss.device)
                # loss is float32 because ``mse_loss`` layers ``autocast`` to float32
                assert recon_loss.dtype is torch.float32
            """
            x = x.to(device)
            print("x.device: ", x.device)
            x_hat, commitment_loss, codebook_loss, perplexity = model(x, epoch)
            print("x_hat.device: ", x_hat.device)
            recon_loss = mse_loss(x_hat, x)
            print("recon_loss: ", recon_loss.device)

            # for the first 5 epochs loss == recon_loss, aka. no quantisation, warm up 
            loss = recon_loss + commitment_loss * commitment_beta + codebook_loss
            print(loss)

            # Print gradient norms before scaling
            for name, param in model.named_parameters():
                if param.grad is not None:
                    print(f"Before scaling - {name}: grad norm {param.grad.norm(2)}")

            # Exits ``autocast`` before backward().
            # Backward passes under ``autocast`` are not recommended.
            # Backward ops run in the same ``dtype`` ``autocast`` chose for corresponding forward ops.
            # Scales loss. Calls ``backward()`` on scaled loss to create scaled gradients.
            # scaler.scale(loss).backward()
            loss.backward()

            # Print gradient norms
            for name, param in model.named_parameters():
                if param.grad is not None:
                    print(f"After scaling - {name}: grad norm {param.grad.norm(2)}")
                    
            # scaler.unscale_(optimizer)
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
            
            # Print gradient norms 
            for name, param in model.named_parameters():
                if param.grad is not None:
                    print(f"After gradient clipping - {name}: grad norm {param.grad.norm(2)}")
            
            # ``scaler.step()`` first unscales the gradients of the optimizer's assigned parameters.
            # If these gradients do not contain ``inf``s or ``NaN``s, optimizer.step() is then called,
            # otherwise, optimizer.step() is skipped.
            # scaler.step(optimizer)
            optimizer.step()
            optimizer.zero_grad()

            # Updates the scale for next iteration.
            # scaler.update()
        
            wandb.log({
                'epoch': epoch + 1,
                'batch_idx': batch_idx + 1,
                'recon_loss': recon_loss.item(),
                'commitment_loss': commitment_loss.item(),
                'codebook_loss': codebook_loss.item(),
                'total_loss': loss.item(),
                'perplexity': perplexity.item()
            })
                
            if batch_idx % print_step == 0: 
                print("epoch:", epoch + 1, "(", batch_idx + 1, ") recon_loss:", recon_loss.item(), " perplexity: ", perplexity.item(), 
                    " commit_loss: ", commitment_loss.item(), "\n\t codebook loss: ", codebook_loss.item(), " total_loss: ", loss.item(), "\n")
            
        codebook.random_restart()
        codebook.reset_usage()
    print("Finish!!")
    wandb.finish()


def main():
    # Parse the command-line argument
    parser = argparse.ArgumentParser()
    parser.add_argument('--wandb_key', type=str, required=True)
    parser.add_argument('--save_model', type=str, required=False)
    args = parser.parse_args()
    os.environ['WANDB_API_KEY'] = args.wandb_key
    os.environ['WANDB_BASE_URL'] = 'https://api.wandb.ai'
    os.environ['WANDB_USER_EMAIL'] = 'anna.klezovich24@gmail.com'

    device = str('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_default_device(device)

    train_args = {
        'batch_size': 30,
        'img_size': (32, 32),  # (width, height) # NOT USED
        'input_dim': 112,
        'hidden_dim': 512,
        'latent_dim': 16,
        'n_embeddings': 512,
        'output_dim': 112,
        'commitment_beta': 0.30,
        'lr': 2e-4,
        'epochs': 200,  # 50
        'print_step': 50,
        'sequence_length': 30,
        'use_amp': False,
    }
    # Initialize the model
    encoder = arch.Encoder(input_dim=train_args['input_dim'], hidden_dim=train_args['hidden_dim'], output_dim=train_args['latent_dim'])
    # codebook = arch.VQEmbeddingEMA(n_embeddings=train_args['n_embeddings'], embedding_dim=train_args['latent_dim'])
    codebook = ResidualVQ(
            dim = 124,  # latent_dim 16, but in lib it is 256
            codebook_size = train_args['n_embeddings'],
            num_quantizers = 4,
            kmeans_init = True,   # set to True
            kmeans_iters = 10,     # number of kmeans iterations to calculate the centroids for the codebook on init
            stochastic_sample_codes = True,
            sample_codebook_temp = 0.2,         # temperature for stochastically sampling codes, 0 would be equivalent to non-stochastic
            shared_codebook = True,
            #return_loss_breakdown=True,
            #codebook_diversity_loss_weight=0.1
        )
    decoder = arch.Decoder(input_dim=train_args['latent_dim'], hidden_dim=train_args['hidden_dim'], output_dim=train_args['output_dim'])  
    model = arch.ModelResidualVQ(Encoder=encoder, Codebook=codebook, Decoder=decoder).to(device)  # .to(DEVICE, torch.bfloat16)  # mixed precision training

    training_dataset, test_dataset, train_loader, test_loader = load_data(
        './SSL_video_eaf/SSLC_poses_norm.zip', output_dim=train_args['output_dim'],
        normalize_by_mean_pose=False, num_workers=1,  # it's already pre-normalized
        batch_size=30, sequence_length=train_args['sequence_length']
    )

    # log in to wandb relogin
    wandb.login(relogin=True)

    # Initialize W&B
    wandb.init(project="vqvae_on_ssl", name="rqvae")

    mse_loss = nn.MSELoss()
    optimizer = AdamW(model.parameters(), lr=train_args['lr'])

    # Start training
    train(model, optimizer, mse_loss, train_loader, train_args['epochs'], train_args['print_step'], device, wandb, codebook, train_args['sequence_length'], use_amp=train_args['use_amp'])

    # Save the model
    if args.save_model:
        model_name = 'vq_vae_restart-nowarmup-from-zip-noamp-200epochs.pth'  # args.save_model
        torch.save(model.state_dict(), model_name)


if __name__ == "__main__":
    main()
