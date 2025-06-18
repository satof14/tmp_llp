import os
from torchvision.datasets import CIFAR10

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
os.makedirs(DATA_DIR, exist_ok=True)

def save_split(root, split):
    dataset = CIFAR10(root=root, train=(split=='train'), download=True)
    split_dir = os.path.join(root, 'cifar10', split)
    for idx, (img, label) in enumerate(dataset):
        label_dir = os.path.join(split_dir, str(label))
        os.makedirs(label_dir, exist_ok=True)
        path = os.path.join(label_dir, f"{idx:05d}.png")
        img.save(path)

if __name__ == "__main__":
    data_root = DATA_DIR
    for split in ['train', 'test']:
        print(f"Preparing CIFAR-10 {split} images...")
        save_split(data_root, split)
    print("CIFAR-10 dataset prepared under data/cifar10/")
