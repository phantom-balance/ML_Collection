import torch
import os.path
import torch.optim as optim
import torchvision.transforms as transforms
from model import CNNtoRNN
from loader import get_loader
from utils import load_checkpoint
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose(
    [
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
)
train_loader, dataset = get_loader(
    root_folder="flickr8k/images",
    captions_file="flickr8k/captions.txt",
    transformation=transform,
    num_workers=2
)

embed_size = 256
hidden_size = 256
vocab_size = len(dataset.vocab)
num_layers = 1
learning_rate = 0.0003


model = CNNtoRNN(embed_size, hidden_size, vocab_size, num_layers).to(device)
optimizer = optim.Adam(model.parameters(),lr=learning_rate)


def img_cap(root_dir=None, img_id_list=None):
    model.eval()
    load_checkpoint(torch.load("IMG_CAP.pth.tar", map_location=device), model, optimizer)

    for idx, img_id in enumerate(img_id_list):
        img_pth = os.path.join(root_dir, img_id)
        img = Image.open(img_pth).convert("RGB")
        img = transform(img)
        # print(img.shape)
        # print(img.unsqueeze(0).shape)
        img = img.unsqueeze(0)
        tokens = model.caption_image(img.to(device), dataset.vocab)
        print(f"{img_id}:image{idx}:{tokens}")
    model.train()


# img_id_list = ["doggo.jpg","Important.png"]
# img_cap(root_dir="Image", img_id_list=img_id_list)

