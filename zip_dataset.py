import io
import zipfile
from pathlib import Path
import pandas as pd

import numpy as np
from tqdm import tqdm
import torch

from utils import *
import importlib
import utils
utils = importlib.reload(utils)

sslc_pose = pd.read_csv('sslc_pose_2.csv', encoding='utf-8')

poses_dir = './SSL_video_eaf/SSLC_poses/'
sequence_length = 30


def save_poses_to_zip(directory: str, zip_filename: str, normalize_by_mean_pose=True):
    # pose_files = list(Path(directory).glob("*.pose"))
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        sslc_pose_dataset = utils.PoseDataset(
            df=sslc_pose,
            root_dir=poses_dir,
            sequence_length=sequence_length,
            normalize_by_mean_pose=True
        )
        
        for data_tensor, filename in tqdm(sslc_pose_dataset, total=len(sslc_pose_dataset)):
            # pose = pre_process_mediapipe(pose)
            # pose = normalize_mean_std(pose)

            # Using the file name as the zip entry name
            npz_filename = filename + '.npz'

            # Saving the masked array to a temporary buffer
            with io.BytesIO() as buf:
                # data = pose.body.data[:, 0, :, :]  # only first person

                float16_data = data_tensor.to(torch.float16)
                # np.set_printoptions(precision=10, floatmode="fixed")
                np.savez_compressed(buf, data=float16_data)
                zip_file.writestr(npz_filename, buf.getvalue())


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Preprocess data')
    parser.add_argument('--dir', type=str, help='Directory containing the pose files')
    parser.add_argument('--out', type=str, help='Output zip file')
    parser.add_argument('--norm', type=bool, help='Normalize by mean pose')

    args = parser.parse_args()

    save_poses_to_zip(args.dir, args.out, normalize_by_mean_pose=True)   # python3 zip_dataset.py --dir "./SSL_video_eaf/SSLC_poses/" --out "./SSL_video_eaf/SSLC_poses.zip" --norm True