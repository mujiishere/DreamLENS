import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import clip
import sys, os

# Add the parent directory (DreamLENS) to the system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from eeg_dataset import EEGClipDataset
 
# Load CLIP model
clip_model, preprocess = clip.load("ViT-B/32", device='cpu')

# Load your dataset
data_path = r"C:\Amrita\S5 DL PROJECT DATASET\EEG-ImageNet_1.pth"   # Change this to your dataset file
dataset = EEGClipDataset(data_path, clip_model=clip_model, preprocess=preprocess, mode='train', granularity='both')

# ---- Basic Stats ----
print("\n--- Dataset Summary ---")
print(f"Total samples: {len(dataset)}")
labels = [item['label'] for item in dataset]
unique_labels = set(labels)
print(f"Unique labels: {len(unique_labels)}")
subjects = [item['subject'] for item in dataset]
print(f"Unique subjects: {len(set(subjects))}")

# ---- Visualize Random EEG Signal ----
idx = np.random.randint(0, len(dataset))
sample = dataset[idx]

plt.figure(figsize=(10, 4))
plt.plot(sample['eeg'].numpy().T)
plt.title(f"EEG Signal — Label: {sample['label']} | Subject: {sample['subject']}")
plt.xlabel("Time Steps")
plt.ylabel("EEG Channel Amplitude")
plt.show()

# ---- Visualize CLIP Embedding Space (t-SNE) ----
# Get embeddings and labels for a subset
subset_size = min(200, len(dataset))
embeddings = torch.stack([dataset[i]['target_embedding'] for i in range(subset_size)]).numpy()
labels_subset = [dataset[i]['label'] for i in range(subset_size)]

# t-SNE projection to 2D
tsne = TSNE(n_components=2, random_state=42)
embeddings_2d = tsne.fit_transform(embeddings)

# Plot
plt.figure(figsize=(8, 6))
scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                      c=[hash(lbl) % 10 for lbl in labels_subset], cmap='tab10', alpha=0.7)
plt.title("t-SNE of CLIP Text Embeddings for EEG Labels")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.show()

# ---- EEG Length/Channel Distribution ----
lengths = [sample['eeg'].shape[-1] for sample in dataset]
channels = [sample['eeg'].shape[0] for sample in dataset]

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.hist(lengths, bins=20)
plt.title("EEG Signal Length Distribution")

plt.subplot(1, 2, 2)
plt.hist(channels, bins=20)
plt.title("EEG Channel Count Distribution")
plt.show()
