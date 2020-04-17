import os
import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import skimage.transform

import torch
import torch.nn as nn
import torchvision.transforms as transforms

import models


def caption(net:nn.Module, image:Image.Image, device:torch.device, beam_searcher: models.BeamSearcher):
    # image_resized = transforms.Resize(256, 256)(image)
    image_tensor = transforms.Compose((
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ))(image)

    image_tensor = image_tensor.unsqueeze(0)
    image_tensor = image_tensor.to(device)

    net.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        captions, scores, attention_weights = net.predict(image_tensor, beam_searcher)
    
    caption   = captions[0]
    score     = scores[0]
    attention = attention_weights[0]

    attention = attention.resize(len(caption), 14, 14)

    return caption, score, attention


def visualize_att(image_path, seq, alphas, rev_word_map, smooth=True):
    """
    Visualizes caption with weights at every word.

    Adapted from paper authors' repo: https://github.com/kelvinxu/arctic-captions/blob/master/alpha_visualization.ipynb

    :param image_path: path to image that has been captioned
    :param seq: caption
    :param alphas: weights
    :param rev_word_map: reverse word mapping, i.e. ix2word
    :param smooth: smooth weights?
    """
    image = Image.open(image_path)
    image = image.resize([14 * 24, 14 * 24], Image.LANCZOS)

    words = [rev_word_map[ind] for ind in seq]

    for t in range(len(words)):
        if t > 50:
            break
        plt.subplot(np.ceil(len(words) / 5.), 5, t + 1)

        plt.text(0, 1, '%s' % (words[t]), color='black', backgroundcolor='white', fontsize=12)
        plt.imshow(image)
        current_alpha = alphas[t, :]
        if smooth:
            alpha = skimage.transform.pyramid_expand(current_alpha.numpy(), upscale=24, sigma=8)
        else:
            alpha = skimage.transform.resize(current_alpha.numpy(), [14 * 24, 14 * 24])
        if t == 0:
            plt.imshow(alpha, alpha=0)
        else:
            plt.imshow(alpha, alpha=0.8)
        plt.set_cmap(cm.Greys_r)
        plt.axis('off')
    plt.show()


def my_visualize_att(image, words, alphas, smooth=True):
    """
    Visualizes caption with weights at every word.

    Adapted from paper authors' repo: https://github.com/kelvinxu/arctic-captions/blob/master/alpha_visualization.ipynb

    :param image_path: path to image that has been captioned
    :param seq: caption
    :param alphas: weights
    :param rev_word_map: reverse word mapping, i.e. ix2word
    :param smooth: smooth weights?
    """
    image = image.resize([14 * 24, 14 * 24], Image.LANCZOS)

    alphas = alphas.detach().cpu().numpy()
    alphas = [np.zeros_like(alphas[0])] + list(alphas)
    words = ['<bos>'] + words

    for t in range(len(words)):
        if t > 50:
            break
        plt.subplot(np.ceil(len(words) / 5.), 5, t + 1)

        plt.text(0, 1, '%s' % (words[t]), color='black', backgroundcolor='white', fontsize=12)
        plt.imshow(image)
        current_alpha = alphas[t]
        if smooth:
            alpha = skimage.transform.pyramid_expand(current_alpha, upscale=24, sigma=8)
        else:
            alpha = skimage.transform.resize(current_alpha, [14 * 24, 14 * 24])
        if t == 0:
            plt.imshow(alpha, alpha=0)
        else:
            plt.imshow(alpha, alpha=0.8)
        plt.set_cmap(cm.Greys_r)
        plt.axis('off')
    plt.show()


if __name__ == "__main__":

    # dataset_guide_path = './data_config/CocoTestDatasetGuide.json'
    checkpoint_path    = './checkpoints/checkpoint_best.pth'
    # checkpoint_path    = './checkpoints2/checkpoint_best.pth'
    word_map_path      = './data_config/DefaultWordMap.json'
    # word_map_path      = './data_config/SelectGloveWordMap.json'
    # image_path         = 'D:\\hhcaz\\ZJU\\Junior_spring_summer\\Machine Learning\\datasets\\MSCOCO14\\data_origin\\val2014\\COCO_val2014_000000000073.jpg'
    image_path         = 'D:\\hhcaz\\ZJU\\Junior_spring_summer\\Machine Learning\\datasets\\VizWiz_Captions\\val\\VizWiz_val_00000548.jpg'

    device = torch.device("cuda:0")

    with open(word_map_path, 'r') as j:
        word_map = json.load(j)
    inv_word_map = {v:k for k, v in word_map.items()}
    
    beam_searcher = models.BeamSearcher(word_map['<bos>'], word_map['<eos>'], beam_size=5, max_steps=50, device=device)
    checkpoint = torch.load(checkpoint_path)
    net = checkpoint['model']

    image = Image.open(image_path).convert('RGB')

    net.to(device)
    caption, score, attention = caption(net, image, device, beam_searcher)
    caption = [inv_word_map[idx] for idx in caption]

    print(len(caption), score, attention.shape)

    print(' '.join(caption))

    my_visualize_att(image, caption, attention)

