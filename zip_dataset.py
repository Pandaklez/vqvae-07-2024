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


# Corpus dataframe path: 'sslc_pose_2.csv'
#sslc_pose = pd.read_csv('./lexicon_jsons/sign_data_frames_info_train.pkl', encoding='utf-8')

# Corpus jsons address: './SSL_video_eaf/SSLC_poses/' 
poses_dir = './lexicon_jsons/out/'
sequence_length = 30


def save_poses_to_zip(poses_dir: str, zip_filename: str, sslc_pose: str, normalize_by_mean_pose=True):
    # pose_files = list(Path(poses_dir).glob("*.pose"))
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        sslc_pose_dataset = utils.PoseDataset2(
            df=sslc_pose,
            root_dir=poses_dir,
            sequence_length=sequence_length,
            normalize_by_mean_pose=normalize_by_mean_pose,
            trim=False,  # trim should be False for lexicon and True for corpus
        )
        
        for data_tensor, filename in tqdm(sslc_pose_dataset, total=len(sslc_pose_dataset)):
            # pose = pre_process_mediapipe(pose)
            # pose = normalize_mean_std(pose)

            # Using the file name as the zip entry name
            npz_filename = filename + '.npz'
            float16_data = data_tensor.cpu().numpy().astype(np.float16)
            # float16_data = data_tensor.to(torch.float16)

            # Saving the masked array to a temporary buffer
            with io.BytesIO() as buf:
                np.set_printoptions(precision=10, floatmode="fixed")
                np.savez_compressed(buf, data=float16_data)
                buf_value = buf.getvalue()
                zip_file.writestr(npz_filename, buf_value)
            print(f"Saved {npz_filename} to {zip_filename}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Preprocess data')
    parser.add_argument('--dir', type=str, help='Directory containing the pose files')
    parser.add_argument('--out', type=str, help='Output zip file')
    parser.add_argument('--df', type=str, help='Dataframe containing the pose files')
    parser.add_argument('--norm', type=bool, help='Normalize by mean pose')

    args = parser.parse_args()

    # Lexicon dataframe path: 'sign_data_frames_info_train.pkl', 'sign_data_frames_info_test.pkl', 'sign_data_frames_info_val.pkl'
    sslc_pose = pd.read_pickle(args.df)
    # Corpus dataframe path: 'sslc_pose_2.csv'
    # sslc_pose = pd.read_csv(args.df, encoding='utf-8')

    save_poses_to_zip(args.dir, args.out, sslc_pose, normalize_by_mean_pose=args.norm) 

    # corpus: python3 zip_dataset.py --dir "./SSL_video_eaf/SSLC_poses/" --out "./SSL_video_eaf/SSLC_poses.zip" --norm True
    # lexicon: python zip_dataset.py --dir "./lexicon_jsons/out/" --out "./lexicon_jsons/lexicon_poses_norm_train2.zip" --df "./lexicon_jsons/sign_data_frames_info_train.pkl" --norm True
    # lexicon: python zip_dataset.py --dir "./lexicon_jsons/out/" --out "./lexicon_jsons/lexicon_poses_norm_test2.zip" --df "./lexicon_jsons/sign_data_frames_info_test.pkl" --norm True
    # lexicon: python zip_dataset.py --dir "./lexicon_jsons/out/" --out "./lexicon_jsons/lexicon_poses_norm_val2.zip" --df "./lexicon_jsons/sign_data_frames_info_val.pkl" --norm True