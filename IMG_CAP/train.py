import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from utils import save_checkpoint, load_checkpoint
from loader import get_loader
from model import CNNtoRNN


def train():
    transform = transforms.Compose(
        [
            transforms.Resize((356, 356)),
            transforms.RandomCrop((299,299)),
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    load_model = False
    save_model = True

    embed_size = 256
    hidden_size = 256
    vocab_size = len(dataset.vocab)
    num_layers = 1
    learning_rate = 0.0003
    num_epochs = 100

    model = CNNtoRNN(embed_size, hidden_size, vocab_size, num_layers).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.vocab.stoi["<PAD>"])
    optimizer = optim.Adam(model.parameters(),lr=learning_rate)

    if load_model:
        load_checkpoint(torch.load("IMG_CAP.pth.tar", map_location=device), model, optimizer)

    model.train()

    for epoch in range(num_epochs):
        if save_model:
            checkpoint ={
                "state_dict" : model.state_dict(),
                "optimizer" : optimizer.state_dict()
            }
            save_checkpoint(checkpoint)

# I dont fully understand this yet
        for idx, (imgs, captions) in enumerate(train_loader):
            imgs = imgs.to(device)
            captions = captions.to(device)

            outputs = model(imgs, captions[:-1])
            loss = criterion(outputs.reshape(-1, outputs.shape[2]), captions.reshape(-1))

            optimizer.zero_grad()
            loss.backward(loss)
            optimizer.step()


if __name__ == "__main__":
    train()
