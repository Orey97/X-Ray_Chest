import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from PIL import Image

class ChestXRayDataset(Dataset):
    def __init__(self, dataframe, image_dir, transform=None):
        self.dataframe = dataframe.reset_index(drop=True) #riassegnamo gli indici da 0 a n-1 dopo lo split
        self.image_dir = image_dir
        self.transform = transform #pipeline di trasformazioni (Resize, ToTensor, Normalize, …).

        self.image_path = self.dataframe["image"].values
        #image non è più una label e patient serve solo allo split già fatto non al training
        self.labels_col = [col for col in self.dataframe.columns if col not in ["image", "patientid"]]
        #Creiamo matrice dopo l'one-hot-encoding, ogni riga avrà valori 0/1 per ogni label
        self.labels = self.dataframe[self.labels_col].values.astype('float32')
        #Necessario convertire in float32 per PyTorch

    def __len__(self):
        return len(self.dataframe) #serve a pytorch quando fermarsi 
    
    def __getitem__(self, idx): #idx sta per posizione(index)
        img_name = self.image_path[idx]
        img_path = os.path.join(self.image_dir, img_name)

        #caricare immagine e convertire in RGB (ray di solito sono scala di grigi invece CNN si aspetta 3 canali)
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)
        '''
        trasformiamo la nostra nostra PIL image in un tensore PyTorch
        fa 1) Resize in 224x224 se non già fatto 2) Augmentation (leggere variazioni delle immagini come rotazioni, flip orizzontali, zoom) 
        3) Converte in tensore da Height x Width x Channels a Channels x Height x Width , perchè Pythorc usa CHW
        4) Normalizza tensor = (tensor - mean) / std per ogni canale RGB
        '''
        labels = torch.tensor(self.labels[idx], dtype=torch.float32) #convertiamo array numpy in tensore PyTorch
        return image, labels
    

def get_transforms(image_size=224):
    train_transform = transforms.Compose([ #tranform per il training set
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=5),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  #sono valori standard per ImageNet
            std=[0.229, 0.224, 0.225]
        )
    ])
        
    base_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)), #transform per validation e test set
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    return train_transform, base_transform, base_transform


def create_dataloader(train_df, val_df, test_df, image_dir, batch_size=32, num_workers=0, image_size=224):
    train_tf, val_tf, test_tf = get_transforms(image_size)

    train_set = ChestXRayDataset(train_df, image_dir, transform=train_tf)
    val_set = ChestXRayDataset(val_df, image_dir, transform=val_tf)
    test_set = ChestXRayDataset(test_df, image_dir, transform=test_tf)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    #pin memory ottimizza performance quando si passa dati dalla CPU alla GPU

    return train_loader, val_loader, test_loader 

