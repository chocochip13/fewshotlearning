import numpy as np
from PIL import Image

from torch.utils.data import Dataset
from torch.utils.data.sampler import BatchSampler
from torchvision import datasets, transforms
import pandas as pd
import torch



class CUHK_train(Dataset):
    def __init__(self,train_path,train):
        # Transforms
        self.to_tensor = transforms.ToTensor()
        self.transform=None
        # Read the csv file
        self.train=train
        if self.train:
            self.train_data_info = pd.read_csv(train_path,header=None)
            self.train_data =[] 
            
            print("printing train data length CUHK")
            print(len(self.train_data_info.index))

            for (i,j) in enumerate(np.asarray(self.train_data_info.iloc[:, 1])):
                try :
                    img = Image.open(j)
                    p = transforms.Compose([transforms.Grayscale(num_output_channels=1),transforms.Resize((256,256))])   
                    self.train_data.append(self.to_tensor(p(img)))

                    #self.train_data.append(self.to_tensor(Image.open(j)))
                except : 
                    print(j)
            

            self.train_data = torch.stack(self.train_data)
            self.train_labels = np.asarray(self.train_data_info.iloc[:, 2], dtype=np.int32)
            #self.train_labels = torch.from_numpy(self.train_labels)
            
            random_state = np.random.RandomState(13)


            self.labels_set = set(self.train_labels)
            self.label_to_indices = {label.item(): np.where(np.asarray(self.train_labels) == label)[0]
                                     for label in self.labels_set}




            self.train_data_len = len(self.train_data_info.index)

            

    def __getitem__(self, index):
        if self.train:
            img1, label1 = self.train_data[index], self.train_labels[index].item()
            positive_index = index
            while positive_index == index:
                positive_index = np.random.choice(self.label_to_indices[label1])
            negative_label = np.random.choice(list(self.labels_set - set([label1])))
            negative_index = np.random.choice(self.label_to_indices[negative_label])
            img2 = self.train_data[positive_index]
            img3 = self.train_data[negative_index]
        return (img1, img2, img3), []



    def __len__(self):
        if self.train :
            return self.train_data_len

        
class CUHK_test(Dataset):
    def __init__(self,test_path,test):
        # Transforms
        self.to_tensor = transforms.ToTensor()
        self.transform=None
        # Read the csv file
        self.test=test
        if self.test:
            self.test_data_info = pd.read_csv(test_path,header=None)
            self.test_data =[] 
            for (i,j) in enumerate(np.asarray(self.test_data_info.iloc[:, 1])):
                try :
                    img = Image.open(j)
                    p = transforms.Compose([transforms.Grayscale(num_output_channels=1),transforms.Resize((256,256))])
                    self.test_data.append(self.to_tensor(p(img)))
                    #self.test_data.append(self.to_tensor(Image.open(j))) 
                except : 
                    print(j)  

            self.test_data = torch.stack(self.test_data)
            self.test_labels = np.asarray(self.test_data_info.iloc[:, 2], dtype=np.int32)
            #self.test_labels = torch.from_numpy(self.test_labels)
            
            self.test_data_len = len(self.test_data_info.index)
            
            self.labels_set = set(self.test_labels)
                        
            self.label_to_indices = {label.item(): np.where(self.test_labels == label)[0]
                                     for label in self.labels_set}

            random_state = np.random.RandomState(13)

            # print(self.label_to_indices)

            triplets = []
            for i in range(len(self.test_data)):
                    triplets.append([i,
                                random_state.choice(self.label_to_indices[self.test_labels[i].item()]),
                                random_state.choice(self.label_to_indices[
                                                        np.random.choice(
                                                            list(self.labels_set - set([self.test_labels[i].item()]))
                                                        )
                                                    ])
                                ])
            self.test_triplets = triplets
            

    def __getitem__(self, index):
        if self.test:
            img1 = self.test_data[self.test_triplets[index][0]]
            img2 = self.test_data[self.test_triplets[index][1]]
            img3 = self.test_data[self.test_triplets[index][2]]
        return (img1, img2, img3), []


    def __len__(self):
        if self.test :
            return self.test_data_len
