from torch.utils.data import Dataset, DataLoader
import numpy as np

class NumberDataset(Dataset):
    def __init__(self, imgs_path, labels_path, split="train"):
        print("Loading dataset...")

        self.imgs = np.load(imgs_path)
        self.labels = np.load(labels_path)
        self.split = split

        print("Dataset loaded.")

        start_index = 0
        end_index = 0

        if(self.split == "train"):
            start_index = 0
            end_index = int(0.8*len(self.imgs))
        elif(self.split == "val"):
            start_index = int(0.8*len(self.imgs))
            end_index = len(self.imgs)
        elif(self.split == "test"):
            start_index = 0
            end_index = len(self.imgs)

        # *按照比例把数据集分成训练集、验证集和测试集
        self.imgs_list = self.imgs[start_index:end_index]
        self.labels_list = self.labels[start_index:end_index]

        print("Dataset split: ", self.split)
        print("Shape of images: ", self.imgs_list.shape) 
        
    def __getitem__(self, index):
        img = self.imgs_list[index]
        label_idx = self.labels_list[index]

        label_one_hot = self.one_hot(label_idx)

        # *数据归一化
        img = img/255.0

        return img, label_one_hot
    def __len__(self):
        return len(self.labels_list)
    
    def one_hot(self, label):
        res = np.zeros(10)
        res[label] = 1
        return res   
    
    

if __name__ == '__main__':
    train_dataset = NumberDataset("archive/train_imgs.npy", "archive/train_labels.npy", "train")
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    for i, (img, label) in enumerate(train_loader):
        print(img[0][0][13])
        print(label[0])
        break

    val_dataset = NumberDataset("archive/train_imgs.npy", "archive/train_labels.npy", "val")
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)

    for i, (img, label) in enumerate(val_loader):
        print(img.shape)
        print(label.shape)
        break

    test_dataset = NumberDataset("archive/test_imgs.npy", "archive/test_labels.npy", "test")
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    for i, (img, label) in enumerate(test_loader):
        print(img.shape)
        print(label.shape)
        break
