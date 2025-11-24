import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageOps
import pandas as pd
import numpy as np
import os
import re
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.ndimage import label, binary_dilation

# Set up logging for PyTorch to reduce console noise during operations
# import logging
# logging.getLogger('torch').setLevel(logging.ERROR)


class DoublePositive:
    """
    Processes 3D image stacks (double-channel fluorescence) to identify,
    cluster, filter, and quantify double-positive signals.

    Replaces slow recursive and iterative Python loops with fast, vectorized
    operations using PyTorch and SciPy.
    """

    def __init__(self, path: str, positive_threshold: float = 0.05, cluster_threshold: int = 30):
        """
        Initializes the DoublePositive processor.

        Args:
            path (str): Base directory containing "Channel 2", "Channel 3", and "Results.csv".
            positive_threshold (float): Intensity threshold (fraction of max) for positivity.
            cluster_threshold (int): Minimum size for a connected component to be kept.
        """
        self.path = Path(path)
        self.positive_threshold = positive_threshold
        self.cluster_threshold = cluster_threshold

        # Initialized data structures
        self.channel_1_tensor = None
        self.channel_2_tensor = None
        self.label_points = None
        self.raw_double_positive = None
        self.denoised_double_positive = None
        self.keypoint_mask = None
        self.final_mask = None
        self.image_shape = None

        # Statistics
        self.total_volume = 0
        self.total_volume_weighted = 0
        self.total_count = 0
        self.background_threshold = 1000  # Default background threshold

    def _natural_sort_key(self, s):
        """Helper for natural sorting of filenames."""
        s = str(s)
        return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

    def load_images(self, file_path: Path) -> torch.Tensor:
        """Loads image sequence from a directory, sorts naturally, and stacks into a 3D tensor."""
        file_list = [item for item in file_path.iterdir() if not item.is_dir()]
        file_list.sort(key=self._natural_sort_key)

        tensor_list = []
        to_tensor = transforms.ToTensor()
        for file in file_list:
            img = Image.open(file)
            img = ImageOps.grayscale(img)
            img_tensor = to_tensor(img)
            tensor_list.append(img_tensor)

        if not tensor_list:
            raise FileNotFoundError(f"No images found in {file_path}")

        tensor_list = torch.stack(tensor_list)
        print(f'Loaded Images Shape from {file_path.name}: {tensor_list.shape}')
        return tensor_list

    def load_label(self) -> torch.Tensor:
        """Loads labeled points from Results.csv."""
        df = pd.read_csv(self.path / "Results.csv", usecols=[7, 8])
        return torch.tensor(df.values)

    def init_data(self):
        """Loads channels, calculates raw double positive signal, and initializes masks."""
        print("Loading image data...")
        self.channel_1_tensor = self.load_images(self.path / "Channel 2")
        self.channel_2_tensor = self.load_images(self.path / "Channel 3")

        if (self.path / "Results.csv").exists():
            self.label_points = self.load_label()
        else:
            self.label_points = None

        # Calculate raw double positive: element-wise multiplication
        # Assumes channels are (Z, 1, H, W) -> Result is (Z, H, W)
        self.raw_double_positive = self.channel_1_tensor * self.channel_2_tensor
        self.raw_double_positive = self.raw_double_positive.squeeze(1)

        self.image_shape = self.raw_double_positive.shape
        print(f'Image Stack Shape: {self.image_shape}')

    def denoise(self):
        """
        Applies intensity thresholding and a specific 3D de-noising filter.
        The filter keeps pixels only if they have neighbors in the Z-axis (i-1 and i+1).
        """
        print("De-noising and Thresholding...")
        # 1. Intensity Thresholding
        max_val = torch.max(self.raw_double_positive)
        min_val = torch.min(self.raw_double_positive)
        threshold_val = min_val + (max_val - min_val) * self.positive_threshold

        # Create a binary mask based on the intensity threshold
        raw_positive_mask = (self.raw_double_positive > threshold_val).float()

        # Update the raw signal to zero out anything below threshold
        self.raw_double_positive[raw_positive_mask == 0] = 0

        # 2. 3D Convolutional Denoising (Checking Z-axis neighbors)
        # Kernel: [1, 0, 1] along the Z (temporal) dimension
        denoise_kernel = torch.ones((3, 1, 1), dtype=torch.float32)
        denoise_kernel[1, 0, 0] = 0
        denoise_kernel = denoise_kernel.view(1, 1, 3, 1, 1) # (out_channels, in_channels, D, H, W)

        # Pad and reshape mask for 3D convolution: (1, 1, D, H, W)
        conv_input = raw_positive_mask.unsqueeze(0).unsqueeze(0)

        # Perform convolution
        # We use 'replicate' padding to handle edges in the Z dimension
        denoised_conv = F.conv3d(
            conv_input, denoise_kernel, padding=(1, 0, 0)
        )
        denoised_conv = denoised_conv.squeeze()

        # Denoised mask: only pixels where both z-neighbors were positive (conv result == 2)
        denoised_mask = (denoised_conv == 2).float()

        # Apply the final denoised mask to the raw signal
        self.denoised_double_positive = self.raw_double_positive * denoised_mask

    def generate_mask(self):
        """
        VECTORIZED: Finds connected components (clusters) in the denoised signal
        using scipy.ndimage.label, which is much faster than recursive Python calls.
        """
        print("Generating mask via Connected Component Labeling...")
        # Convert the denoised tensor to a NumPy binary array for fast CCL
        binary_data = (self.denoised_double_positive > 0).numpy().astype(int)

        # Perform 3D Connected Component Labeling (using 3x3x3 connectivity by default)
        # The output 'labels' is the keypoint_mask, 'num_labels' is self.label - 1
        labels, num_labels = label(binary_data)
        
        # Convert back to torch tensor
        self.keypoint_mask = torch.from_numpy(labels).to(torch.int32)
        self.label = num_labels + 1
        self.final_mask = self.keypoint_mask.clone()

    def remove_small_clusters(self):
        """
        VECTORIZED: Calculates the size of each cluster and removes those below
        the cluster_threshold.
        """
        print("Removing small clusters...")
        # Get counts for each label (0 is background, ignore)
        self.keypoint_counts = torch.bincount(self.keypoint_mask.view(-1))

        # Indices of clusters to keep (size >= threshold AND label > 0)
        # Index 0 is background, which we always ignore in counts.
        self.selected_idx_to_keep = torch.where(self.keypoint_counts >= self.cluster_threshold)[0]
        self.selected_idx_to_keep = self.selected_idx_to_keep[self.selected_idx_to_keep > 0] # Exclude background (label 0)

        # Create a new mask where only the kept indices remain
        # This is a highly efficient way to filter tensor values
        final_mask_np = self.keypoint_mask.numpy()
        mask_filter = np.isin(final_mask_np, self.selected_idx_to_keep.numpy())
        final_mask_np[~mask_filter] = 0
        self.final_mask = torch.from_numpy(final_mask_np).to(torch.int32)

    def readd_points(self):
        """
        VECTORIZED: Re-adds raw positive pixels that are directly adjacent (3D)
        to the boundaries of the kept clusters (dilation/expansion).
        """
        print("Re-adding raw positive points adjacent to kept clusters (Dilation)...")

        # 1. Create a binary mask of the *kept* clusters
        kept_mask_np = (self.final_mask > 0).numpy().astype(bool)

        # 2. Dilate the kept mask by 1 unit in 3D
        # This expands the mask to include the immediate 3D neighbors
        # Structure is the default 3x3x3 connectivity
        dilated_mask_np = binary_dilation(kept_mask_np, structure=np.ones((3, 3, 3)))

        # 3. Identify the pixels added by dilation (the new boundary points)
        new_boundary_np = dilated_mask_np & (~kept_mask_np)

        # 4. Check which of these new boundary points are also in the RAW positive signal
        raw_positive_np = (self.raw_double_positive > 0).numpy().astype(bool)
        points_to_readd_np = new_boundary_np & raw_positive_np

        # 5. Assign the original cluster label to the points being re-added
        # We check the original keypoint_mask (before small cluster removal)
        # to get the correct label for the re-added point.
        readd_indices = np.argwhere(points_to_readd_np)
        
        keypoint_mask_np = self.keypoint_mask.numpy()
        final_mask_np = self.final_mask.numpy()

        for i, j, k in readd_indices:
            # Check the original label of the raw point
            original_label = keypoint_mask_np[i, j, k]
            
            # Since the raw point should be adjacent to a kept cluster, 
            # we check its neighbors in the FINAL mask to assign a label.
            # This is safer than relying on the (possibly removed) original label.
            # We use a 3x3x3 neighborhood check on the current final_mask
            
            # Find the label of the closest neighbor in the final_mask
            min_i, max_i = max(0, i - 1), min(self.image_shape[0], i + 2)
            min_j, max_j = max(0, j - 1), min(self.image_shape[1], j + 2)
            min_k, max_k = max(0, k - 1), min(self.image_shape[2], k + 2)
            
            # Get unique labels in the neighbor region of the final mask, excluding 0
            neighbor_labels = np.unique(final_mask_np[min_i:max_i, min_j:max_j, min_k:max_k])
            valid_neighbor_labels = neighbor_labels[neighbor_labels > 0]

            if valid_neighbor_labels.size > 0:
                # Assign the first found valid neighbor label (arbitrary choice if multiple)
                final_mask_np[i, j, k] = valid_neighbor_labels[0]

        self.final_mask = torch.from_numpy(final_mask_np).to(torch.int32)
        self.final_points_counts = torch.bincount(self.final_mask.view(-1))


    def count_statistics(self):
        """Calculates total count, total volume (count-based), and total volume (weighted)."""
        print("Counting statistics...")
        
        # Determine background indices (clusters larger than background_threshold, usually one large object)
        self.background_idx_list = []
        
        for idx in range(1, self.final_points_counts.shape[0]):
            count = self.final_points_counts[idx].item()
            if count < self.background_threshold:
                # Signal cluster
                self.total_volume += count
                self.total_count += 1
            else:
                # Background cluster (e.g., large artifact or cell body)
                self.background_idx_list.append(idx)

        # Create a binary mask of ONLY the counted signal (excluding background cluster indices)
        final_mask_binary = torch.ones_like(self.final_mask).float()
        final_mask_binary[self.final_mask == 0] = 0 # Remove global background (label 0)
        
        # Remove identified background clusters
        for idx in self.background_idx_list:
            final_mask_binary[self.final_mask == idx] = 0

        # Calculate weighted volume: sum of raw intensities within the final signal mask
        self.total_volume_weighted = torch.sum(self.raw_double_positive * final_mask_binary).item()

        print(f'Total Count: {self.total_count}')
        print(f'Total Volume (count-based): {self.total_volume}')
        print(f'Total Volume (weighted): {self.total_volume_weighted:.2f}')

    def save_statistics(self):
        """Saves calculated statistics to a text file."""
        output_path = self.path / "statistics.txt"
        with open(output_path, 'w') as f:
            f.write(f'Image Shape: {self.image_shape}\n')
            f.write(f'Total Count: {self.total_count}\n')
            f.write(f'Total Volume (count-based): {self.total_volume}\n')
            f.write(f'Total Volume (raw/weighted): {self.total_volume_weighted:.2f}\n')
            
            pixel_area = self.image_shape[1] * self.image_shape[2]
            f.write(f'Total Volume normalized (count-based): {self.total_volume / pixel_area:.4f}\n')
            f.write(f'Total Volume normalized (raw/weighted): {self.total_volume_weighted / pixel_area:.4f}\n')
        print(f"Statistics saved to {output_path}")

    def save_data(self, file_path: str):
        """Saves intermediate tensor data to a file."""
        data = {
            "raw_double_positive": self.raw_double_positive,
            "denoised_double_positive": self.denoised_double_positive,
            "keypoint_mask": self.keypoint_mask,
            "final_mask": self.final_mask,
            "image_shape": self.image_shape,
            "total_count": self.total_count,
            "total_volume": self.total_volume,
            "total_volume_weighted": self.total_volume_weighted,
        }
        torch.save(data, file_path)
        print(f"Data saved to {file_path}")
        
    def load_data(self, file_path: str):
        """Loads intermediate tensor data from a file."""
        data = torch.load(file_path)
        self.raw_double_positive = data["raw_double_positive"]
        self.denoised_double_positive = data["denoised_double_positive"]
        self.keypoint_mask = data["keypoint_mask"]
        self.final_mask = data["final_mask"]
        self.image_shape = data["image_shape"]
        self.total_count = data["total_count"]
        self.total_volume = data["total_volume"]
        self.total_volume_weighted = data["total_volume_weighted"]
        print(f"Data loaded from {file_path}")

    def process(self):
        """Runs the complete processing pipeline."""
        print("--- Double Positive Signal Processing ---")
        self.init_data()
        self.denoise()
        self.generate_mask()
        self.remove_small_clusters()
        self.readd_points()
        self.count_statistics()
        self.save_statistics()
        print("--- Processing Complete ---")

    def plot_3d(self):
        """
        Plots the final mask and label points in 3D.
        Uses vectorized coordinate extraction for speed.
        """
        if self.final_mask is None:
             print("Run .process() or .load_data() before plotting.")
             return

        print("Preparing 3D plot data...")
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # VECTORIZED EXTRACTION of coordinates for the final signal mask
        # Non-zero returns a tensor of (N, 3) where N is the number of points
        coords = torch.nonzero(self.final_mask, as_tuple=False).to(torch.float32)
        labels = self.final_mask[self.final_mask != 0]
        intensities = self.raw_double_positive[self.final_mask != 0]

        # Axes: z=coords[:, 0], y=coords[:, 1], x=coords[:, 2] (swapped for typical viewing)
        # Note: Added a factor of 0.1 to z-axis for visualization scale, as in original code
        x = coords[:, 2].numpy()
        y = coords[:, 1].numpy()
        z = (coords[:, 0] * 0.1 + 0.1).numpy()
        c = labels.numpy()
        s = (intensities * 10).numpy() # Scale marker size by intensity

        if self.label_points is not None and self.label_points.numel() > 0:
            # Add user-labeled points to the plot data (Z=0)
            label_x = self.label_points[:, 1].numpy()
            label_y = self.label_points[:, 0].numpy()
            label_z = np.zeros_like(label_x)
            label_c = np.full_like(label_x, np.max(c) + 1 if c.size > 0 else 1) # Use a distinct color index
            label_s = np.full_like(label_x, 50) # Fixed size for labels

            x = np.concatenate([x, label_x])
            y = np.concatenate([y, label_y])
            z = np.concatenate([z, label_z])
            c = np.concatenate([c, label_c])
            s = np.concatenate([s, label_s])

        if x.size == 0:
            print("No positive signals found to plot.")
            return

        print(f"Plotting {x.size} points...")
        # Use a colormap that is distinct from the label color for the user points
        cmap = plt.cm.get_cmap('viridis')
        
        # Normalize colors based on max label index
        norm_c = plt.Normalize(vmin=1, vmax=np.max(c))
        
        sc = ax.scatter(x, y, z, c=c, cmap=cmap, s=s, norm=norm_c)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('3D Double Positive Signal Clusters')
        plt.colorbar(sc, label='Cluster ID')
        plt.show()

if __name__ == "__main__":
    dp = DoublePositive(path="/home/yue/Desktop/UpState/DoublePositive/data/Set 4/Group A",positive_threshold=0.05,cluster_threshold=20)
    dp.process()
    dp.plot_3d()