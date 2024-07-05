import json
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
from itertools import islice
import os
from torch.utils.data import Dataset, IterableDataset
import torch
import math
from random import randrange, seed
from pathlib import Path
import random
import time
import zipfile
import io


def load_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def get_full_sequence_length(data):
    sequence_length = len(data.keys())
    return sequence_length

def draw_bodypose_json(body, ax, size=16, width=2, draw_all=False):
    candidate = np.array(body['candidate'])
    subset = np.array(body['subset'])

    limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], \
               [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17], \
               [1, 16], [16, 18], [3, 17], [6, 18]]

    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

    for i in range(17):
        for n in range(len(subset)):
            index = subset[n][np.array(limbSeq[i]) - 1]
            if -1 in index and not draw_all:
                continue
            X = candidate[index.astype(int), 0]  
            Y = (candidate[index.astype(int), 1]) 
            ax.plot(X, Y, color=np.array(colors[i])/255.0, linewidth=width)
            

    for i in range(18):
        for n in range(len(subset)):
            index = int(subset[n][i])
            if index == -1 and not draw_all:
                continue
            x = candidate[index][0]
            y = (candidate[index][1])
            ax.scatter(x, y, color=np.array(colors[i])/255.0, s=size)

    return 

def draw_bodypose(body, ax):

    if body.shape[0] == 14:
        body = add_unused_joints(body)
    limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], \
               [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17], \
               [1, 16], [16, 18], [3, 17], [6, 18]]

    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]
    
    for i in range(body.shape[0]):
        if i not in [9, 10, 12, 13]:
            x = body[i][0]
            y = body[i][1]
            if i in [4,7]:
                size = 100
                #print(f"Drawing joint {i} at {x}, {y}")
            else:
                size = 4
            ax.scatter(x, y, color=np.array(colors[i])/255.0, s=size)

        if i < body.shape[0]-1 and i not in [7, 8, 10, 11]:
            X = body[limbSeq[i][0]-1][0], body[limbSeq[i][1]-1][0]  
            Y = body[limbSeq[i][0]-1][1], body[limbSeq[i][1]-1][1]
            ax.plot(X, Y, color=np.array(colors[i])/255.0, linewidth=2)
        
    return 

def draw_handpose(all_hand_peaks, ax, size=4, width=1):
    edges = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [0, 9], [9, 10], \
             [10, 11], [11, 12], [0, 13], [13, 14], [14, 15], [15, 16], [0, 17], [17, 18], [18, 19], [19, 20]]

    for peaks in all_hand_peaks:
        peaks = np.array(peaks)

        for ie, e in enumerate(edges):
            x1, y1 = peaks[e[0]]
            x2, y2 = peaks[e[1]]
            ax.plot([x1, x2], [y1, y2], color=plt.cm.hsv(ie / float(len(edges))), linewidth=width)

        for i, keypoint in enumerate(peaks):
            if i == 0:
                size = 50
                #print(f"Drawing hand at {keypoint}")
            else:
                size = 0
            x, y = keypoint
            ax.scatter(x, y, color='red', s=size)

def get_center(frame):
    candidate = np.array(frame['bodies']['candidate'])
    center = [candidate[1][0], candidate[1][1]]
    return center

def get_scale(frame):
    scale_factor = frame['bodies']['candidate'][5][0]-frame['bodies']['candidate'][2][0]  # measuring x distance between left and right shoulder
    return scale_factor

def scale_bodypose(body, scale_factor):
    body = body / scale_factor
    return body

def shift_bodypose(body, center):
    body = body - center
    return body

def normalize_mean_std(body):
    mean, std = load_mean_and_std()
    #mean_t, std_t = torch.from_numpy(np.array(mean)), torch.from_numpy(np.array(std))
    #print("body.size(), mean_t.size(): ", body.size(), mean_t.size())
    body = (body - mean) / std
    return body

def unnormalize_mean_std(body):
    mean, std = load_mean_and_std()
    # mean_t, std_t = torch.from_numpy(np.array(mean)), torch.from_numpy(np.array(std))
    body = (body * std) + mean
    return body

def correct_aspect_ratio_body(body, aspect_ratio):
    ys = body[:,1]
    out_ys = ys/aspect_ratio
    body_out = np.array([body[:,0], out_ys]).transpose(1,0)
    return body_out

def correct_aspect_ratio_hands(hands, aspect_ratio):
    hands_out = []
    for hand in hands:
        ys = hand[:,1]
        out_ys = ys/aspect_ratio
        hand_out = np.array([hand[:,0],out_ys]).transpose(1,0)
        hands_out.append(hand_out)
    return hands_out

def scale_handpose(hands, scale_factor):
    scaled_hands = []
    for hand in reversed(hands):
        peaks = np.array(hand)
        scaled_hands.append(peaks / scale_factor)
    return scaled_hands

def shift_handpose(hands, center, body, center_on_wrist):
    shifted_hands = []
    for hand, wrist_location in zip(reversed(hands), [body[4], body[7]]):
        # We center the hand based on the under arm end joint location
        if center_on_wrist:
            center = wrist_location
        # Else we use the center provided which is typically from the first frame of the video
        else:
            center = center
        shifted_hands.append(shift_single_handpose(hand, center))
    hands = shifted_hands
    return hands

def reattach_hand(hand, center):
    hand = hand + center
    return hand

def reattach_hands_to_arms(hands, body):
    reattached_hands = []
    for hand, center in zip(reversed(hands), [body[4], body[7]]):
        reattached_hands.append(reattach_hand(hand, center))
    return reattached_hands

def shift_single_handpose(hand, center):
    peaks = np.array(hand)
    shifted_hand = peaks - center
    return shifted_hand

def draw_from_tensor(tensor, ax, reattach_hands=True):
    tensor = tensor.reshape(-1, 2)
    bodies = tensor[:14]
    hands = [tensor[14:35], tensor[35:]]
    if reattach_hands:
        hands = reattach_hands_to_arms(hands, bodies)
    
    draw_bodypose(bodies, ax)
    draw_handpose(hands, ax)

def remove_unused_joints(body_joints):
    out = []
    for i, joint in enumerate(body_joints):
        if i not in [13, 12, 10, 9]:
            out.append(joint)
    return out

def add_unused_joints(body_joints):
    for i in [9, 10, 12, 13]:
        body_joints = np.insert(body_joints, i, [i*0.1,i*0.1], axis=0)
    return body_joints

def frame_to_tensor(frame, center, scale_factor, aspect_ratio, center_on_wrist):
    body = np.array(frame['bodies']['candidate'])
    hands = np.array(frame['hands'])
    hands = scale_handpose(correct_aspect_ratio_hands(shift_handpose(hands, center, body, center_on_wrist=center_on_wrist), aspect_ratio), scale_factor)
    body = remove_unused_joints(scale_bodypose(correct_aspect_ratio_body(shift_bodypose(body, center), aspect_ratio), scale_factor))
    all_data = np.concatenate((body, hands[0], hands[1]), axis=0)

    return all_data  # (56,2)

def concat_frames(frames):
    data = torch.cat(frames, dim=0)
    return data

def find_outlier(tensor):
    t_prev = None
    for i in range(tensor.shape[1]):
        t  = tensor[:,i]
        result = np.zeros_like(t)
        if t_prev is not None:
            diff_arr = t-t_prev
            result[diff_arr > 0.9] = 1
            if np.any(result>0):
                return True
        t_prev = t
    return False
    
def find_outlier_new(tensor):
    diffs = np.diff(tensor, axis=1)
    outliers = np.any(diffs > 0.9, axis=0)
    return np.any(outliers)

def trim(data):
    # trim first 110 frames and last 40 (based on diffs between two coordintes of centers plot)
    frame_keys = sorted(data.keys(), key=lambda x: int(x.split('_')[1]))
    frames_to_delete = frame_keys[:110] + frame_keys[-40:]
    # Delete the unwanted frames from the dictionary
    for frame in frames_to_delete:
        del json_data[frame]
    return json_data

def delete_start_end_frames(json_data, max_frame_number):
    keys_to_delete = [f"frame_{i}" for i in range(1, max_frame_number+1) if i < 111 or i > max_frame_number-40]
    for key in keys_to_delete:
        del json_data[key]
    return json_data

def recursive_append(list1, list2):
    result = []
    def flatten(lst):
        for element in lst:
            if isinstance(element, list):
                flatten(element)  # Recursive call for nested list
            else:
                result.append(element)
    flatten(list1)
    flatten(list2)
    return result

def load_mean_and_std():
    import json
    with open(Path(__file__).parent / "pose_normalization_lexicon.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # TODO: [:21] is a fuck up from mean pose file creation that needs to be fixed separately
    mean = recursive_append(data["bodies"]["candidate"]["mean"], recursive_append(data["hands"]["mean"][0][:21], data["hands"]["mean"][1][:21]))
    std = recursive_append(data["bodies"]["candidate"]["std"], recursive_append(data["hands"]["std"][0][:21], data["hands"]["std"][1][:21]))
    # print("len(mean), len(std) in load_mean_and_std(): ", len(mean), len(std))
    mean = [float(el) for el in mean]
    std = [float(el) for el in std]
    
    # when std is 0, set std to 1
    std = np.array(std)
    std[std == 0] = 1
    return np.array(mean), std

class PoseDataset(Dataset):
    """
    # TODO: 
    1. (Wed) Separate func for calculating mean & std values for each joint and storing it in a json file
    2. (Wed) A func to load this mean & std file
    3. (Wed) A func to do pose = pose - mean / std - for each frame -> This one is instead of `shift_bodypose()` to the center, mean pose is sort of a new center
    4. (Wed) Turn the dataset itself into huggingface dataset so it doesn't run 10 minutes on each run. Or zip just to make things faster in general
    5. (?) Add faces data
    """
    def __init__(self, df, root_dir, sequence_length=30, center_on_wrist=True, random_seed=None, normalize_by_mean_pose=False, trim=True):
        self.df = df
        self.root_dir = root_dir
        self.sequence_length = sequence_length
        self.center_on_wrist = center_on_wrist
        self.normalize_by_mean_pose = normalize_by_mean_pose
        self.current_counter = 0
        self.length = len(self.df)
        self.trim = trim
        print(self.trim)
        if random_seed is not None: 
            seed(random_seed)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx=None):
        skipped = 0
        if idx is None:
            idx = randrange(0, len(self.df))
        while True:
            try:
                row = self.df.iloc[idx]
                filename = os.path.join(self.root_dir, row['movie_filename'].split('.')[0] + str('.json'))
            except Exception as e:
                skipped += 1
                #idx += 1
                #print(f"At df index skipped")
                continue
            try:
                json_data = load_json_file(filename)
            except Exception as e:
                skipped += 1
                #print(f"At json reading, Skipped {skipped} times")
                #print(e)
                #idx += 1
                continue

            # print("filename: ", filename)
            
            # trim first 110 frames and last 40
            max_frame_number = get_full_sequence_length(json_data)

            if self.trim:
                json_data = delete_start_end_frames(json_data, max_frame_number)
            
            aspect_ratio = row['aspect_ratio']
            frame_rate = int(row['frame_rate'])
            #print(f"Frame rate: {frame_rate}, Aspect ratio: {aspect_ratio}")
            resampling_factor = round(frame_rate/25)
            #print(f"Resampling factor: {resampling_factor}")
            
            if self.trim:
                # finding center of first frame
                center = get_center(json_data['frame_112'])
                # scaling all frames to have the same scale
                scale_factor = get_scale(json_data['frame_112'])
            else:
                # finding center of first frame
                center = get_center(json_data['frame_1'])
                # scaling all frames to have the same scale
                scale_factor = get_scale(json_data['frame_1'])
            
            # Sampling a possible start frame        
            start_frame = randrange(0, max_frame_number - self.sequence_length*resampling_factor)
            stop_frame = start_frame + self.sequence_length * resampling_factor
            data = []
            counter = 0
            for key, frame in islice(json_data.items(), start_frame, stop_frame):
                if counter % resampling_factor == 0:
                    all_data = frame_to_tensor(frame, center, scale_factor, aspect_ratio, center_on_wrist=self.center_on_wrist)

                    if all_data.shape[0] != 56:
                        skipped += 1
                        #idx += 1
                        continue

                    if self.normalize_by_mean_pose:
                        all_data = normalize_mean_std(all_data.flatten()).reshape(56, 2)
                    data.append(all_data)
                    
                counter += 1
            if len(data) != self.sequence_length:
                skipped += 1
                #print("filename: ", filename)
                #print(f"At sequence, Skipped {skipped} times")
                #idx += 1
                continue
            # print(data.shape) # (sequence_length=30,56,2)
            data = np.stack(data).astype(np.float32)  #, dtype=np.float32)
            # print(data.dtype)  # This should not be 'object'
            data_tensor = torch.from_numpy(data)  # torch.tensor(data, dtype=torch.float32)
            data_tensor = data_tensor.reshape(self.sequence_length, -1)
            data_tensor = data_tensor.transpose(0,1)
            
            if find_outlier(data_tensor):
                skipped += 1
                #print("filename: ", filename)
                #print(f"At outliers, Skipped {skipped} times")
                #idx += 1
                continue
            break
        print(f"Returning sample but skipped {skipped} times")

        return data_tensor, row['movie_filename'].split('.')[0]

    def __next__(self):
        if self.current_counter >= self.length:
            raise StopIteration
        self.current_counter += 1
        return self.__getitem__(idx=self.current_counter - 1)

    def __iter__(self):
        return self


def crop_pose(tensor, max_length: int):
    if max_length is not None:
        offset = random.randint(0, len(tensor) - max_length) \
            if len(tensor) > max_length else 0
        return tensor[offset:offset + max_length]
    return tensor


class PackedDataset(IterableDataset):
    def __init__(self, dataset: Dataset, max_length: int, shuffle=True):
        self.dataset = dataset
        self.max_length = max_length
        self.shuffle = shuffle

    def __iter__(self):
        dataset_len = len(self.dataset)
        datum_idx = 0

        datum_shape = self.dataset[0].shape
        # padding_shape = tuple([10] + list(datum_shape)[1:])
        # padding = MaskedTensor(tensor=torch.zeros(padding_shape), mask=torch.zeros(padding_shape))

        while True:
            poses = []
            total_length = 0
            while total_length < self.max_length:
                if self.shuffle:
                    datum_idx = random.randint(0, dataset_len - 1)
                else:
                    datum_idx = (datum_idx + 1) % dataset_len

                # Append pose
                pose = self.dataset[datum_idx]
                poses.append(pose)
                total_length += len(pose)

                # Append padding
                # poses.append(padding)
                # total_length += len(padding)

            concatenated_pose = torch.cat(poses, dim=0)[:self.max_length]  # MaskedTorch.cat(poses, dim=0)[:self.max_length]
            yield concatenated_pose


class _ZipPoseDataset(Dataset):
    def __init__(self, zip_path,   # zip_obj: zipfile.ZipFile,
                 files: list,
                 max_length: int = 512,
                 in_memory: bool = False,
                 dtype=torch.float32,
                 df=None
                ):
        self.max_length = max_length
        self.zip_path = zip_path
        self.files = files
        self.df = df
        self.in_memory = in_memory
        self.dtype = dtype
        self.memory_files = []

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # print("len(self.memory_files), len(self.files):  ", len(self.memory_files), len(self.files))
        if len(self.memory_files) == len(self.files):
            tensor = self.memory_files[idx]
        else:
            while True:
                zipf = zipfile.ZipFile(self.zip_path, 'r')
                
                # If we want to store in memory, we first load sequentially all the files
                idx = idx if not self.in_memory else len(self.memory_files)

                with zipf.open(self.files[idx]) as file:
                    file_content = file.read()  # Read the entire file content
    
                # Convert the bytes content to a BytesIO object and load with numpy
                pose_file = io.BytesIO(file_content)
                pose_array = np.load(pose_file, mmap_mode='r')
                
                # row = self.df.iloc[idx]
                # filename = row['movie_filename'].split('.')[0] + '.npz'
                # print("filename in getitem: ", filename)
                pose_array = pose_array['data']
                try:
                    tensor = torch.from_numpy(pose_array).to(dtype=self.dtype)
                except KeyError:
                    print(f'Filename %s does not exist, skipping' % filename)
                    zipf.close()
                    continue
                # print("Tensor to store in memory:\n", tensor)
                
                # This line used to turn tensor into a masked tensor: tensor = preprocess_pose(pose, dtype=self.dtype)
                if self.in_memory:
                    self.memory_files.append(tensor)
                    # print("len(self.memory_files):  ", len(self.memory_files))
                    if len(self.memory_files) == len(self.files):
                        # print(self.memory_files)  # self.memory_files: List of Tensors
                        break
                    if len(self.memory_files) % 10000 == 0:
                        print_memory()
                zipf.close()
        return crop_pose(tensor, self.max_length)

    def slice(self, start, end):
        return _ZipPoseDataset(zip_path=self.zip_path, files=self.files[start:end],
                               max_length=self.max_length, in_memory=self.in_memory, dtype=self.dtype, df=self.df)



class ZipPoseDataset(_ZipPoseDataset):
    def __init__(self, zip_path: Path, max_length: int = 512, in_memory: bool = False, dtype=torch.float32, df=None):
        print(f"ZipPoseDataset @ {zip_path} with max_length={max_length}, in_memory={in_memory}")

        # pylint: disable=consider-using-with
        # self.zip_obj = zipfile.ZipFile(zip_path, 'r')
        with zipfile.ZipFile(zip_path, 'r') as zip_obj:
            files = zip_obj.namelist()
            print("Total files", len(files))

            super().__init__(zip_path=zip_path, files=files, max_length=max_length, in_memory=in_memory, dtype=dtype, df=df)

        
# Plotting videos in jupyter
def draw_triple_sample_video(n, ax1, ax2, ax3, sequence_length, sample_id, x, x_hat_model1, x_hat_model2, model1_name, model2_name):
    ax1.clear()
    ax2.clear()
    ax3.clear()
    print(f'frame {n} of {sequence_length} for sample video {sample_id}')
    clear_output(wait=True)
    artists = []
    tensor = x[sample_id].cpu()
    
    # Plot the first data
    draw_from_tensor(tensor[:, n], ax1)
    ax1.set_xlim(-2, 2)
    ax1.set_ylim(2, -2)
    ax1.set_aspect('equal')
    ax1.set_title('Original')

    
    tensor2 = x_hat_model1[sample_id].cpu().detach()

    # Plot the second data
    draw_from_tensor(tensor2[:, n], ax2)
    ax2.set_xlim(-2, 2)
    ax2.set_ylim(2, -2)
    ax2.set_aspect('equal')
    ax2.set_title(model1_name)

    tensor3 = x_hat_model2[sample_id].cpu().detach()
    
    # Plot the second data
    draw_from_tensor(tensor3[:, n], ax3)
    ax3.set_xlim(-2, 2)
    ax3.set_ylim(2, -2)
    ax3.set_aspect('equal')
    ax3.set_title(model2_name)

    return artists

def draw_double_sample_video(n, ax1, ax2, sequence_length, sample_id, x, x_hat_model1, model1_name):
    ax1.clear()
    ax2.clear()
    print(f'frame {n} of {sequence_length} for sample video {sample_id}')
    clear_output(wait=True)
    artists = []
    tensor = x[sample_id].cpu()
    
    # Plot the first data
    draw_from_tensor(tensor[:, n], ax1)
    ax1.set_xlim(-2, 2)
    ax1.set_ylim(2, -2)
    ax1.set_aspect('equal')
    ax1.set_title('Original')

    tensor2 = x_hat_model1[sample_id].cpu().detach()

    # Plot the second data
    draw_from_tensor(tensor2[:, n], ax2)
    ax2.set_xlim(-2, 2)
    ax2.set_ylim(2, -2)
    ax2.set_aspect('equal')
    ax2.set_title(model1_name)

    return artists

def draw_sample_video(n, ax1, sequence_length, sample_id, x, name):
    ax1.clear()
    print(f'frame {n} of {sequence_length} for sample video {sample_id}')
    clear_output(wait=True)
    artists = []
    tensor = x[sample_id].cpu().detach()
    
    # Plot the first data
    draw_from_tensor(tensor[:, n], ax1)
    ax1.set_xlim(-2, 2)
    ax1.set_ylim(2, -2)
    ax1.set_aspect('equal')
    ax1.set_title(name)

    return artists