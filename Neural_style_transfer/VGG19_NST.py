import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.utils import save_image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model = models.vgg19(pretrained=True).features
# print(model)


class VGG(nn.Module):
    def __init__(self):
        super().__init__()

        self.chosen_features = ['0', '5', '10', '19', '28']
        self.model = models.vgg19(pretrained=True).features[:29]

    def forward(self, x):
        features = []

        for layer_num, layer in enumerate(self.model):
            x = layer(x)

            if str(layer_num) in self.chosen_features:
                features.append(x)

        return features


model = VGG().to(device=device).eval()


def load_image(image_name):
    image = Image.open(image_name)
    image = loader(image).unsqueeze(0)
    return image.to(device=device)


image_size = 356
# image_size = 200
loader = transforms.Compose(
    [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ]
)


def img_create(original_img, style_img, alpha, beta):
    original_img = load_image(original_img)
    style_img = load_image(style_img)
    print("STYLE", style_img.shape)
    print("ORIGINAL", original_img.shape)
    generated = original_img.clone().requires_grad_(True)

    total_steps = 6000
    learning_rate = 0.003
    optimizer = optim.Adam([generated], lr=learning_rate)

    for step in range(total_steps):
        generated_features = model(generated)
        original_img_features = model(original_img)
        style_features = model(style_img)

        style_loss = content_loss = 0

        for gen_feature, orig_feature, style_feature in zip(
            generated_features, original_img_features, style_features
        ):
            batch_size, channel, height, width = gen_feature.shape
            content_loss += torch.mean((gen_feature - orig_feature)**2)

            # Compute the Gram Matrix
            G = gen_feature.view(channel, height*width).mm(
                gen_feature.view(channel, height*width).t()
            )

            S = style_feature.view(channel, height*width).mm(
                style_feature.view(channel, height*width).t()
            )

            style_loss += torch.mean((G-S)**2)

        total_loss = alpha*content_loss + beta*style_loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if step % 50 == 0:
            print(total_loss)
            save_image(generated, f"Image/generated{step}.png")


# img_create(original_img="Image/Content.jpg", style_img="Image/Style.jpg", alpha=0.5, beta=1)

