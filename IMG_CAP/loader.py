import os
import pandas as pd
import spacy # text to feature
import torch
from torch.nn. utils.rnn import pad_sequence # same seq_len
from torch.utils.data import DataLoader, Dataset
from PIL import Image # Load img
import torchvision.transforms as transforms

spacy_eng = spacy.blank("en")


class Vocabulary:
    def __init__(self, freq_threshold):
        self.freq_threshold = freq_threshold
        self.itos = {0:"<PAD>", 1:"<SOS>", 2:"<EOS>", 3:"<UNK>"}
        self.stoi = {"<PAD>":0, "<SOS>":1, "<EOS>":2, "<UNK>":3}

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenizer_eng(text):
        return [tok.text.lower() for tok in spacy_eng.tokenizer(text)]

    def build_vocabulary(self, sentence_list):
        frequencies = {}
        idx = 4

        for sentence in sentence_list:
            for word in self.tokenizer_eng(sentence):
                if word not in frequencies:
                    frequencies[word]=1
                else:
                    frequencies[word]+=1
                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

    def featurize(self, text):
        tokenized_text = self.tokenizer_eng(text)

        return [
          self.stoi[token] if token in self.stoi else self.stoi["<UNK>"]
          for token in tokenized_text
        ]


class FlickrDataset(Dataset):
    def __init__(self, root_dir, captions_file, transformation=None, freq_threshold=5):
        self.root_dir = root_dir
        self.df = pd.read_csv(captions_file)
        self.transform = transformation
        self.images_id = self.df["image"]
        self.captions = self.df["caption"]

        # Initialize vocabulary and build vocab
        self.vocab = Vocabulary(freq_threshold)
        self.vocab.build_vocabulary(self.captions.tolist())

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        caption = self.captions[index]
        img_id = self.images_id[index]
        img = Image.open(os.path.join(self.root_dir, img_id)).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        featurized_caption = [self.vocab.stoi["<SOS>"]]
        featurized_caption += self.vocab.featurize(caption)
        featurized_caption.append(self.vocab.stoi["<EOS>"])

        return img, torch.tensor(featurized_caption)


class MyCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        imgs = [item[0].unsqueeze(0) for item in batch]
        imgs = torch.cat(imgs, dim=0)
        targets = [item[1] for item in batch]
        targets = pad_sequence(targets, batch_first=False, padding_value=self.pad_idx)

        return imgs, targets


def get_loader(root_folder, captions_file, transformation, batch_size=32, num_workers=2, shuffle=True, pin_memory=True):
    dataset = FlickrDataset(root_folder, captions_file, transformation=transformation)

    pad_idx = dataset.vocab.stoi["<PAD>"]

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
        collate_fn=MyCollate(pad_idx=pad_idx)
    )

    return loader, dataset


# transform = transforms.Compose(
#     [
#         transforms.Resize((224,224)),
#         transforms.ToTensor(),
#         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#     ]
# )

# dataloader,dataset = get_loader(root_folder="flickr8k/images", captions_file="flickr8k/captions.txt", transform=transform)
#
# # print(Vocabulary.itos)
# for idx, (imgs,captions) in enumerate(dataloader):
#     # print(imgs.shape)
#     print(captions.shape)
#     print(captions)
#     # print(for captions in )
