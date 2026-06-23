import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class PairedMicroscopyDataset(Dataset):
    def __init__(self, green_dir, magenta_dir, size=512):
        self.green_dir = green_dir
        self.magenta_dir = magenta_dir

        #get filenames from green dir, sort so pairing is consistent
        self.filenames = sorted([f for f in os.listdir(green_dir) if f.endswith(".png")])

        #resize + convert to tensor in [-1,1] range (what SD expects)
        self.transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),  #[0,1]
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  #[-1,1]
        ])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        green_fname   = self.filenames[idx]
        magenta_fname = green_fname.replace("chA", "chB")  #same file, different channel suffix

        green   = Image.open(os.path.join(self.green_dir,   green_fname)).convert("RGB")
        magenta = Image.open(os.path.join(self.magenta_dir, magenta_fname)).convert("RGB")

        return self.transform(green), self.transform(magenta)

if __name__ == "__main__":
    GREEN_DIR   = "data/microscopy_lora_green"
    MAGENTA_DIR = "data/microscopy_lora_magenta"

    dataset = PairedMicroscopyDataset(GREEN_DIR, MAGENTA_DIR)
    loader  = DataLoader(dataset, batch_size=2, shuffle=True)

    #quick sanity check
    green_batch, magenta_batch = next(iter(loader))
    print("green:", green_batch.shape)    #should be [2, 3, 512, 512]
    print("magenta:", magenta_batch.shape)