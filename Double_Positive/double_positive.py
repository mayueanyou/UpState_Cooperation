import os,sys,torch,re,argparse
import pandas as pd
from pathlib import Path
from torchvision import transforms
from PIL import Image,ImageOps 
import matplotlib.pyplot as plt
import torch.nn.functional as F
from tqdm import tqdm
import tkinter as tk
from tkinter import messagebox

torch.set_printoptions(precision=None, threshold=10000000000, edgeitems=None, linewidth=1000000000, profile=None, sci_mode=None)
sys.setrecursionlimit(200000)

class DoublePositive:
    def __init__(self,path,positive_threshold=0.05,cluster_threshold=30):
        self.path = path
        
        self.total_volume = 0
        self.total_volume_weighted = 0
        self.total_count = 0
        
        self.background_threshold = 1000
        self.background_idx_list = []
        
        self.cluster_threshold = cluster_threshold
        self.positive_threshold = positive_threshold
    
    def load_images(self,file_path):
        def natural_sort_key(s):
            s = str(s)
            return [int(text) if text.isdigit() else text.lower()for text in re.split(r'(\d+)', s)]
        
        path = Path(file_path)
        file_list = []
        for item in path.iterdir():
            if not item.is_dir(): file_list.append(item)
        file_list.sort(key=natural_sort_key)
        
        tensor_list = []
        to_tensor = transforms.ToTensor()
        for file in file_list:
            img = Image.open(file)
            img = ImageOps.grayscale(img)
            img_tensor = to_tensor(img)
            tensor_list.append(img_tensor)
        tensor_list = torch.stack(tensor_list)
        print('Loaded Images Shape:', tensor_list.shape)
        return tensor_list

    def load_label(self):
        df = pd.read_csv(self.path+"/Results.csv", usecols=[7, 8])
        return torch.tensor(df.values)

    def init_data(self):
        self.channel_1_tensor = self.load_images(self.path+"/Channel 2")
        self.channel_2_tensor = self.load_images(self.path+"/Channel 3")
        
        if os.path.exists(self.path+"/Results.csv"):
            self.label_points = self.load_label()
            
        self.raw_double_positive = self.channel_1_tensor * self.channel_2_tensor
        self.raw_double_positive = self.raw_double_positive.squeeze(1)
        
        self.denoised_double_positive = torch.zeros_like(self.raw_double_positive)
        self.keypoint_mask = torch.zeros_like(self.raw_double_positive).to(torch.int32)
        self.label = 1
        self.keypoint_mask_register = torch.zeros_like(self.raw_double_positive).to(torch.int32)
        self.image_shape = self.raw_double_positive.shape
        print(f'Image Shape: {self.image_shape}')
    
    def save_data(self, file_path):
        data = {
            "raw_double_positive": self.raw_double_positive,
            "denoised_double_positive": self.denoised_double_positive,
            "keypoint_mask": self.keypoint_mask,
            "label": self.label,
            "keypoint_mask_register": self.keypoint_mask_register,
            "image_shape": self.image_shape,
            "final_mask": self.final_mask,
            "total_count": self.total_count,
            "total_volume": self.total_volume,
            "total_volume_weighted": self.total_volume_weighted,
        }
        torch.save(data, file_path)
    
    def load_data(self, file_path):
        data = torch.load(file_path)
        self.raw_double_positive = data["raw_double_positive"]
        self.denoised_double_positive = data["denoised_double_positive"]
        self.keypoint_mask = data["keypoint_mask"]
        self.label = data["label"]
        self.keypoint_mask_register = data["keypoint_mask_register"]
        self.image_shape = data["image_shape"]
        self.final_mask = data["final_mask"]
        self.total_count = data["total_count"]
        self.total_volume = data["total_volume"]
        self.total_volume_weighted = data["total_volume_weighted"]

    def save_statistics(self):
        with open(self.path+"statistics.txt", 'w') as f:
            f.write(f'Image Shape: {self.image_shape}\n')
            f.write(f'Total Count: {self.total_count}\n')
            f.write(f'Total Volume (count-based): {self.total_volume}\n')
            f.write(f'Total Volume (raw): {self.total_volume_weighted}\n')
            f.write(f'Total Volume normalized(count-based): {self.total_volume/(self.image_shape[1]*self.image_shape[2])}\n')
            f.write(f'Total Volume normalized(raw): {self.total_volume_weighted/(self.image_shape[1]*self.image_shape[2])}\n')

    def validate_index(self, i, j, k):
        return (
            0 <= i < self.image_shape[0] and
            0 <= j < self.image_shape[1] and
            0 <= k < self.image_shape[2]
        )
    
    
    def recusive_label(self, i, j, k, label):
        if self.keypoint_mask_register[i][j][k] == 1: return
        self.keypoint_mask_register[i][j][k] = 1
        self.keypoint_mask[i][j][k] = label
        for di in [0, 1]:
            for dj in [-1, 0, 1]:
                for dk in [-1, 0, 1]:
                    if di == 0 and dj == 0 and dk == 0: continue
                    ni, nj, nk = i + di, j + dj, k + dk
                    if self.validate_index(ni, nj, nk):
                        if self.raw_double_positive[ni][nj][nk] != 0 and self.keypoint_mask_register[ni][nj][nk] == 0:
                            self.recusive_label(ni, nj, nk, label)

    def check_surranding_3d(self, mix_mask, i, j, k):
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                for dk in [-1, 0, 1]:
                    if di == 0 and dj == 0 and dk == 0: continue
                    ni, nj, nk = i + di, j + dj, k + dk
                    if self.validate_index(ni, nj, nk):
                        if mix_mask[ni][nj][nk] != 0: return mix_mask[ni][nj][nk]
        return 0

    def denoise(self):
        max_val = torch.max(self.raw_double_positive)
        min_val = torch.min(self.raw_double_positive)
        threashold_val = min_val + (max_val - min_val) * self.positive_threshold
        self.raw_double_positive[self.raw_double_positive < threashold_val] = 0
        
        self.raw_double_positive_mask = torch.zeros_like(self.raw_double_positive)
        self.raw_double_positive_mask[self.raw_double_positive > 0] = 1
        
        denoise_kernel = torch.ones((3, 1, 1))
        denoise_kernel[1][0][0] = 0

        denoised_double_positive_conv = F.conv3d(self.raw_double_positive_mask.unsqueeze(0).unsqueeze(0), denoise_kernel.unsqueeze(0).unsqueeze(0), padding='same')
        denoised_double_positive_conv = denoised_double_positive_conv.squeeze(0).squeeze(0)
        denoised_double_positive_mask = torch.zeros_like(denoised_double_positive_conv)
        denoised_double_positive_mask[denoised_double_positive_conv == 2] = 1
        self.denoised_double_positive = self.raw_double_positive * denoised_double_positive_mask

    
    def generate_mask(self):
        for i in tqdm(range(self.image_shape[0])):
            if torch.sum(self.denoised_double_positive[i]) == 0: continue
            for j in range(self.image_shape[1]):
                if torch.sum(self.denoised_double_positive[i][j]) == 0: continue
                for k in range(self.image_shape[2]):
                    if self.denoised_double_positive[i][j][k] > 0:
                        if self.keypoint_mask_register[i][j][k] == 1: continue
                        else:
                            self.recusive_label(i, j, k, self.label)
                            self.label += 1
        self.final_mask = self.keypoint_mask.clone()
    
    def remove_small_clusters(self):
        self.keypoint_counts = torch.bincount(self.keypoint_mask.view(-1))
        self.selected_idx_to_remove = torch.where(self.keypoint_counts < self.cluster_threshold)[0]
        self.selected_idx_to_keep = torch.where(self.keypoint_counts >= self.cluster_threshold)[0]
        
        for idx in self.selected_idx_to_remove: self.final_mask[self.keypoint_mask == idx] = 0
        
        self.final_points_counts = torch.bincount(self.final_mask.view(-1))


    def readd_points(self):
        while True:
            complete = True
            for i in tqdm(range(self.image_shape[0])):
                if torch.sum(self.raw_double_positive[i]) == 0: continue
                for j in range(self.image_shape[1]):
                    if torch.sum(self.raw_double_positive[i][j]) == 0: continue
                    for k in range(self.image_shape[2]):
                        if self.raw_double_positive[i][j][k] > 0 and self.final_mask[i][j][k] == 0:
                            mask_val = self.check_surranding_3d(self.keypoint_mask, i, j, k)
                            if mask_val in self.selected_idx_to_keep and mask_val != 0:
                                self.final_mask[i][j][k] = mask_val
                                complete = False
            if complete: break
        
        self.final_points_counts = torch.bincount(self.final_mask.view(-1))
    
    def count_statistics(self):
        for i in range(self.final_points_counts.shape[0]):
            if self.final_points_counts[i] < self.background_threshold and self.final_points_counts[i] != 0:
                self.total_volume += self.final_points_counts[i]
                self.total_count += 1
            else:self.background_idx_list.append(i)
        
        final_mask_binary = torch.ones_like(self.final_mask)
        final_mask_binary[self.final_mask == 0] = 0
        for idx in self.background_idx_list: final_mask_binary[self.final_mask == idx] = 0

        self.total_volume_weighted = torch.sum(self.raw_double_positive * final_mask_binary).item()

        print(f'Total Count: {self.total_count}')
        print(f'Total Volume (count-based): {self.total_volume}')
        print(f'Total Volume (weighted): {self.total_volume_weighted}')

    def process(self):
        print("Start processing...")
        self.init_data()
        print("De-noising...")
        self.denoise()
        print("Generating mask...")
        self.generate_mask()
        print("Removing small clusters...")
        self.remove_small_clusters()
        #print("Re-adding points...")
        #self.readd_points()
        print("Counting statistics...")
        self.count_statistics()
    
    def plot_3d(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        x,y,z,c,s = [],[],[],[],[]

        for i in tqdm(range(self.image_shape[0])):
            if torch.sum(self.final_mask[i]) == 0: continue
            for j in range(self.image_shape[1]):
                if torch.sum(self.final_mask[i][j]) == 0: continue
                for k in range(self.image_shape[2]):
                    if self.final_mask[i][j][k] != 0:
                        x.append(j)
                        y.append(k)
                        z.append(i*0.1+0.1)
                        #c.append(dp.raw_double_positive[i][j][k])
                        c.append(self.final_mask[i][j][k])
                        s.append(self.raw_double_positive[i][j][k]*10)
        
        if self.label_points is not None:
            for pos in self.label_points:
                x.append(pos[1])
                y.append(pos[0])
                z.append(0)
                c.append(1.0)
                s.append(50)
        ax.scatter(x, y, z, c=c, cmap='viridis', s=s)
        plt.axis('off')
        plt.show()
    

def main2():
    # set_list = ["Set 1","Set 2","Set 3","Set 4"]
    # group_list = ["Group A","Group B","Group C","Group D","Group E"]
    # for s in tqdm(set_list):
    #     for g in tqdm(group_list):
    #         s = 'Set 4'
    #         g = 'Group A'
    #         print(f'Processing {s} {g} ...')
    #         dp = DoublePositive(path=f"/home/yue/Desktop/UpState/DoublePositive/data/{s}/{g}/",positive_threshold=0.05,cluster_threshold=20)
    #         dp.process()
    #         dp.save_statistics()
    #         dp.plot_3d()
    #         print(f'Finished {s} {g}.\n')
    #         break
    #     break
    dp = DoublePositive(path="/home/yue/Desktop/UpState/DoublePositive/data/Set 4/Group A",positive_threshold=0.05,cluster_threshold=30)
    dp.process()
    dp.save_statistics()    
    dp.plot_3d()

def main(args):
    dp = DoublePositive(path=args.path, positive_threshold=args.positive_threshold, cluster_threshold=args.cluster_threshold)
    dp.process()
    dp.save_statistics()
    dp.plot_3d()


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument('-p','--path', type=str)
    # parser.add_argument('-pt','--positive_threshold', type=float, default=0.05)
    # parser.add_argument('-ct','--cluster_threshold', type=int, default=20)
    # args = parser.parse_args()
    # main(args)
    main2()