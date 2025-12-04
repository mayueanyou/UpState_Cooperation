import os,sys,torch,re,argparse
import pandas as pd
from pathlib import Path
from torchvision import transforms
from PIL import Image,ImageOps 
import matplotlib.pyplot as plt
import torch.nn.functional as F
from tqdm import tqdm

torch.set_printoptions(precision=None, threshold=10000000000, edgeitems=None, linewidth=1000000000, profile=None, sci_mode=None)
sys.setrecursionlimit(200000)

class ConnectedComponentLabeling3D:
    def __init__(self, volume,denoise_volume):
        self.volume = volume
        self.denoise_volume = denoise_volume
        self.depth, self.height, self.width = volume.shape
        self.labels = torch.zeros_like(volume, dtype=torch.int32)
        self.current_label = 1
        self.final_labels = None

    def is_valid(self, x, y, z):
        return 0 <= x < self.depth and 0 <= y < self.height and 0 <= z < self.width

    def dfs(self, x, y, z):
        stack = [(x, y, z)]
        while stack:
            cx, cy, cz = stack.pop()
            if not self.is_valid(cx, cy, cz): continue
            if self.volume[cx, cy, cz] == 0 or self.labels[cx, cy, cz] != 0: continue
            self.labels[cx, cy, cz] = self.current_label
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    for dz in [-1, 0, 1]:
                        if dx == 0 and dy == 0 and dz == 0: continue
                        stack.append((cx + dx, cy + dy, cz + dz))

    def label_components(self,min_size=30, max_size=1000):
        for x in tqdm(range(self.depth)):
            if torch.sum(self.volume[x]) == 0: continue
            for y in range(self.height):
                if torch.sum(self.volume[x][y]) == 0: continue
                for z in range(self.width):
                    if self.denoise_volume[x, y, z] != 0 and self.labels[x, y, z] == 0:
                        self.dfs(x, y, z)
                        self.current_label += 1
                        
        self.remove_out_range_clusters(min_size=min_size, max_size=max_size)
        self.count_statistics()

    def remove_out_range_clusters(self, min_size, max_size):
        label_counts = torch.bincount(self.labels.view(-1))
        for label, count in enumerate(label_counts):
            if count < min_size or count > max_size:
                self.labels[self.labels == label] = 0
    
    def count_statistics(self):
        self.label_counts = torch.bincount(self.labels.view(-1))
        self.total_count = int(self.label_counts[self.label_counts > 0].numel()) - 1
        self.total_volume = torch.sum(self.label_counts[1:]).item()
        
        final_mask_binary = torch.ones_like(self.labels)
        final_mask_binary[self.labels == 0] = 0
        self.total_volume_weighted = torch.sum(self.volume * final_mask_binary).item()

        print(f'Total Count: {self.total_count}')
        print(f'Total Volume (count-based): {self.total_volume}')
        print(f'Total Volume (weighted): {self.total_volume_weighted}')
        self.total_volume_normalized = self.total_volume / (self.height * self.width)
        self.total_volume_weighted_normalized = self.total_volume_weighted / (self.height * self.width)
        print(f'Total Volume normalized (count-based): {self.total_volume_normalized}')
        print(f'Total Volume normalized (weighted): {self.total_volume_weighted_normalized}')
    
    def plot_labels(self, ax,label_points=None):
        x, y, z, c, s = [], [], [], [], []

        for i in tqdm(range(self.depth)):
            if torch.sum(self.labels[i]) == 0: continue
            for j in range(self.height):
                if torch.sum(self.labels[i][j]) == 0: continue
                for k in range(self.width):
                    if self.labels[i][j][k] != 0:
                        x.append(j)
                        y.append(k)
                        z.append(i)
                        c.append(self.labels[i][j][k])
                        s.append(self.volume[i][j][k] * 10)
        
        if label_points is not None:
            for pos in label_points:
                x.append(pos[1])
                y.append(pos[0])
                z.append(0)
                c.append(1.0)
                s.append(50)
        
        # draw a 3D frame (wireframe box) around the image volume
        x_min, x_max = 0, self.height - 1
        y_min, y_max = 0, self.width - 1
        z_min, z_max = 0.0, (self.depth - 1)

        # base and top corners (x corresponds to height index j, y to width index k)
        base = [(x_min, y_min, z_min), (x_max, y_min, z_min), (x_max, y_max, z_min), (x_min, y_max, z_min)]
        top = [(x, y, z_max) for (x, y, _) in base]

        # plot the 12 edges of the box
        for i in range(4):
            # base edge
            ax.plot([base[i][0], base[(i + 1) % 4][0]],
                    [base[i][1], base[(i + 1) % 4][1]],
                    [base[i][2], base[(i + 1) % 4][2]],
                    color='black', linewidth=1)
            # top edge
            ax.plot([top[i][0], top[(i + 1) % 4][0]],
                    [top[i][1], top[(i + 1) % 4][1]],
                    [top[i][2], top[(i + 1) % 4][2]],
                    color='black', linewidth=1)
            # vertical edge
            ax.plot([base[i][0], top[i][0]],
                    [base[i][1], top[i][1]],
                    [base[i][2], top[i][2]],
                    color='black', linewidth=1)

        # set axis limits to frame bounds with a small margin
        ax.set_xlim(x_min - 1, x_max + 1)
        ax.set_ylim(y_min - 1, y_max + 1)
        ax.set_zlim(z_min - 0.1, z_max + 0.1)
        
        ax.scatter(x, y, z, c=c, cmap='viridis', s=s)
        # set view to front-top (top-down from the front)
        ax.view_init(elev=110, azim=0)
        # add axis labels
        ax.set_xlabel('Height (pixels)')
        ax.set_ylabel('Width (pixels)')
        ax.set_zlabel('Depth (slices)')
        
        # reserve space and draw the title text at the bottom of the figure
        ax.set_title(f'Positive Detection Results \n Total Count: {self.total_count},\n Total Volume (count-based): {self.total_volume},\n Total Volume (weighted): {self.total_volume_weighted}')

class PositiveDetection:
    def __init__(self,path,double_positive_threshold=0.05,double_cluster_threshold=35,
                 trible_positive_threshold=0.01,trible_cluster_threshold=20):
        self.path = path
        
        self.label_points = None
        
        self.background_threshold = 1000
        self.background_idx_list = []
        
        self.double_cluster_threshold = double_cluster_threshold
        self.double_positive_threshold = double_positive_threshold
        
        self.trible_cluster_threshold = trible_cluster_threshold
        self.trible_positive_threshold = trible_positive_threshold
        
        self.fig = plt.figure()
        # create two side-by-side 3D subplots by overriding fig.add_subplot so the next two calls create (1,2,1) and (1,2,2)
        self._orig_add_subplot = self.fig.add_subplot
        self._subplot_call_count = 0
        def _add_subplot_override(*args, **kwargs):
            self._subplot_call_count += 1
            if self._subplot_call_count == 1:
                return self._orig_add_subplot(1, 2, 1, **kwargs)
            elif self._subplot_call_count == 2:
                return self._orig_add_subplot(1, 2, 2, **kwargs)
            else:
                return self._orig_add_subplot(*args, **kwargs)
        self.fig.add_subplot = _add_subplot_override
        self.fig.set_size_inches(12, 6)
        self.ax1 = self.fig.add_subplot(111, projection='3d')
        self.ax2 = self.fig.add_subplot(111, projection='3d')
    
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
        self.channel_3_tensor = self.load_images(self.path+"/Channel 1")
        
        if os.path.exists(self.path+"/Results.csv"):
            self.label_points = self.load_label()
            
        self.raw_double_positive = self.channel_1_tensor * self.channel_2_tensor
        self.raw_double_positive = self.raw_double_positive.squeeze(1)
        self.raw_triple_positive = self.raw_double_positive * self.channel_3_tensor.squeeze(1)
        
        self.raw_double_positive, self.denoised_double_positive = self.denoise(self.raw_double_positive, self.double_positive_threshold)
        self.raw_triple_positive, self.denoised_triple_positive = self.denoise(self.raw_triple_positive, self.trible_positive_threshold)
        
        self.image_shape = self.raw_double_positive.shape
        print(f'Image Shape: {self.image_shape}')

    def save_statistics(self):
        with open(self.path+"/statistics.txt", 'w') as f:
            f.write('--- Double Positive Detection Statistics ---\n')
            f.write(f'Double Positive Threshold: {self.double_positive_threshold}\n')
            f.write(f'Double Cluster Threshold: {self.double_cluster_threshold}\n\n')
            f.write(f'Total Count: {self.ccl3d_double.total_count}\n')
            f.write(f'Total Volume (count-based): {self.ccl3d_double.total_volume}\n')
            f.write(f'Total Volume (weighted): {self.ccl3d_double.total_volume_weighted}\n')
            f.write(f'Total Volume normalized (count-based): {self.ccl3d_double.total_volume_normalized}\n')
            f.write(f'Total Volume normalized (weighted): {self.ccl3d_double.total_volume_weighted_normalized}\n\n')
            f.write(f'--- Trible Positive Detection Statistics ---\n')
            f.write(f'Trible Positive Threshold: {self.trible_positive_threshold}\n')
            f.write(f'Trible Cluster Threshold: {self.trible_cluster_threshold}\n')
            f.write(f'Total Count: {self.ccl3d_triple.total_count}\n')
            f.write(f'Total Volume (count-based): {self.ccl3d_triple.total_volume}\n')
            f.write(f'Total Volume (weighted): {self.ccl3d_triple.total_volume_weighted}\n')
            f.write(f'Total Volume normalized (count-based): {self.ccl3d_triple.total_volume_normalized}\n')
            f.write(f'Total Volume normalized (weighted): {self.ccl3d_triple.total_volume_weighted_normalized}\n')

    def denoise(self,raw_data,positive_threshold):
        max_val = torch.max(raw_data)
        min_val = torch.min(raw_data)
        threashold_val = min_val + (max_val - min_val) * positive_threshold
        
        raw_data[raw_data < threashold_val] = 0
        
        raw_data_mask = torch.zeros_like(raw_data)
        raw_data_mask[raw_data > 0] = 1
        
        denoise_kernel = torch.ones((3, 1, 1))
        denoise_kernel[1][0][0] = 0

        denoised_positive_conv = F.conv3d(raw_data_mask.unsqueeze(0).unsqueeze(0), denoise_kernel.unsqueeze(0).unsqueeze(0), padding='same')
        denoised_positive_conv = denoised_positive_conv.squeeze(0).squeeze(0)
        denoised_positive_mask = torch.zeros_like(denoised_positive_conv)
        denoised_positive_mask[denoised_positive_conv == 2] = 1
        denoised_positive = raw_data * denoised_positive_mask
        return raw_data,denoised_positive
    
    def plot_3d(self):
        plt.axis('on')
        plt.show()

    def process(self):
        print("Start processing...")
        self.init_data()
        
        self.ccl3d_double = ConnectedComponentLabeling3D(self.raw_double_positive,self.denoised_double_positive)
        self.ccl3d_double.label_components(self.double_cluster_threshold,self.background_threshold)
        self.ccl3d_double.plot_labels(self.ax1,self.label_points)
        self.ccl3d_triple = ConnectedComponentLabeling3D(self.raw_triple_positive,self.denoised_triple_positive)
        self.ccl3d_triple.label_components(self.trible_cluster_threshold,1000)
        self.ccl3d_triple.plot_labels(self.ax2,self.label_points)
        self.save_statistics()
        print("Processing finished.")
        #self.plot_3d()

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
    dp = PositiveDetection(path="/home/yue/Desktop/UpState/DoublePositive/data/Set 4/Group A")
    dp.process()

def main(args):
    dp = PositiveDetection(path=args.path, positive_threshold=args.positive_threshold, cluster_threshold=args.cluster_threshold)
    dp.process()


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument('-p','--path', type=str)
    # parser.add_argument('-pt','--positive_threshold', type=float, default=0.05)
    # parser.add_argument('-ct','--cluster_threshold', type=int, default=20)
    # args = parser.parse_args()
    # main(args)
    main2()