# This code file generates CSV files from the MNIST dataset and visualizes 
# a shuffled subset of the data as PNG images. The code is structured 
# into three main sections: 
#   - data generation, 
#   - CSV creation, and 
#   - visualization.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.utils import shuffle
from pathlib import Path  # <--- Step 1: Import Path

def generate_mnist_csv(df, n_per_digit, output_filename):

    subsets = []
    print(f"Sampling {n_per_digit} images per digit for '{output_filename}'...")
    for digit in range(10):
        # Filter for the specific digit
        digit_df = df[df['class'].astype(int) == digit]
        # Randomly sample the required number
        sample = digit_df.sample(n=n_per_digit, random_state=42)
        subsets.append(sample)
    
    # Combine and save (keeping them grouped initially for structure)
    final_df = pd.concat(subsets).reset_index(drop=True)
    final_df.to_csv(output_filename, index=False, header=False)
    print(f"File '{output_filename}' saved with shape: {final_df.shape}")

def visualize_shuffled_png(csv_filename, png_output_name):
    """
    Reads the CSV, shuffles the rows, and saves a grid of images to PNG.
    """
    print(f"Reading and Shuffling {csv_filename}")
    # Load the CSV
    data = pd.read_csv(csv_filename, header=None)
    
    # Shuffle the dataframe rows randomly
    # random_state=None will give you a different shuffle every 
    # time you run the script
    data_shuffled = shuffle(data, random_state=None).reset_index(drop=True)
    
    # X: First 784 columns | y: The 785th column (index -1)
    X = data_shuffled.iloc[:, :-1].values 
    y = data_shuffled.iloc[:, -1].values

    # Setup the grid
    rows, cols = 3, 4
    fig, axes = plt.subplots(rows, cols, figsize=(10, 8))
    fig.suptitle("Randomly Shuffled MNIST Samples", fontsize=16)
    
    for i, ax in enumerate(axes.flatten()):
        if i < len(X):
            # Reshape 784 -> (28, 28)
            image_matrix = X[i].reshape(28, 28)
            
            # Using 'magma' or 'gray' colormap
            ax.imshow(image_matrix, cmap='magma')
            ax.set_title(f"Digit: {int(y[i])}")
            ax.axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save the file
    plt.savefig(png_output_name)
    plt.close() # Close plot to free up memory on the cluster
    print(f"Shuffled visualization saved to '{png_output_name}'")

output_dir = Path("./data")
output_dir.mkdir(parents=True, exist_ok=True)

print(20*"=" + "\nFetching MNIST dataset\n" + 20*"=")
mnist = fetch_openml('mnist_784', version=1, as_frame=True, parser='auto')
df = mnist.frame

# Normalize pixel values to [0, 1]
pixel_cols = df.columns.drop('class')
df[pixel_cols] = df[pixel_cols].astype('float64') / 255.0

# Generate the full CSV (if you haven't already)
file_path_full = Path("./data/mnist_full.csv")
if file_path_full.exists():
    print(f"File '{file_path_full}' already exists.")
else:
    df.to_csv('./data/mnist_full.csv', index=False, header=False)

file_path_mini = Path("./data/mnist_mini.csv")
file_path_micro = Path("./data/mnist_micro.csv")
if file_path_mini.exists() and file_path_micro.exists():
    print(f"Files '{file_path_mini}' and '{file_path_micro}' already exist.")
else:
    # Generate the CSV (if you haven't already)
    generate_mnist_csv(df, 600, "./data/mnist_mini.csv")

if file_path_micro.exists():
    print(f"File '{file_path_micro}' already exists. Skipping visualization.")
else:
    # Generate the CSV (if you haven't already)
    generate_mnist_csv(df, 100, "./data/mnist_micro.csv")

# Shuffle, visualize, and save to PNG
visualize_shuffled_png("./data/mnist_micro.csv", "./data/shuffled_digits.png")