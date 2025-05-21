import string
from pathlib import Path

import cv2  # noqa
import nltk
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from PIL import Image


def image2vec_resnet(img_path: Path):
    model = models.resnet18()
    model = torch.nn.Sequential(*(list(model.children()))[:-2])  # 最後の全結合層を削除
    model.eval()

    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

    img = Image.open(img_path).convert("RGB")
    x = transform(img).unsqueeze(0)  # shape: (1, 3, 224, 224)

    with torch.no_grad():
        features = model(x)  # shape: (1, 512, 7, 7)

    patches = features.squeeze().permute(1, 2, 0).reshape(-1, 512)  # shape: (49, 512)
    return patches.numpy()


def doc2vec(text: string):
    nltk.download("punkt")
    nltk.download("stopwords")

    text = text.lower()
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    return tokens


def bag_of_words(tokenized):
    word2id = {}
    id2word = {}
    counter = 0
    for tokens in tokenized:
        for tok in tokens:
            word2id[tok] = counter
            id2word[counter] = tok
            counter += 1
    return len(word2id)
