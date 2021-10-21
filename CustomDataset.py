import torch

class CustomDataset(torch.utils.data.Dataset): 
    def __init__(self):
        bs = 128
        self.x_data = torch.rand([bs,1,28,28])
        self.y_data = torch.randint(0,10,[bs])
    
    def __len__(self): 
        return len(self.x_data)
    
    def __getitem__(self, idx): 
        x = self.x_data[idx]
        y = self.y_data[idx]
        return x, y
    
    def getshape(self):
        size = self.x_data[0].size()
        return list(size)