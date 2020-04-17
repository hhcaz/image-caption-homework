import os
import time
import json
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

import models
from utils import AverageMeter
# from models import ShowAttendTell
from datasets import CaptionDataset


# # debug
# word_map = json.load(open('./data_config/DefaultWordMap.json'))
# idx2word = {v: k for k, v in word_map.items()}

def validate(net: nn.Module, dataloader: DataLoader, device: torch.device):
    total_batches = len(dataloader)

    losses = AverageMeter()
    accs   = AverageMeter()

    references = []
    hypotheses = []

    print('[INFO] Start validating...')

    net.eval()
    with torch.no_grad():
        for i, (image_ids, images, captions, cap_lens, all_captionss) in enumerate(tqdm(dataloader)):
            # image_ids: tuple, shape=(batch_size,)
            # images: shape=(batch_size, 3, H, W)
            # captions: word indexes, shape=(batch_size, cap_len)
            # cap_lens: true caption length, shape=(batch_size,)
            # all_captionss: tuple of list of list, shape=(batch_size, n_catrgories, cap_len)

            # Move to GPU if available
            images   = images.to(device)
            captions = captions.to(device)
            cap_lens = cap_lens.to(device)

            predictions, alphas = net(images, captions, cap_lens)
            # predictions' shape=(batch_size, max_cap_len, vocab_size)
            # alphas' shape=(batch_size, num_pixels)
            # predictions usual end with <eos> but <bos> is excluded
            decoder_cap_lens = cap_lens - 1

            # Record the preditions and ground truths
            pred_idxes = predictions.argmax(dim=2) # shape=(batch_size, max_cap_len)
            for i, pred_idx in enumerate(pred_idxes):
                pred_idx = pred_idx[:decoder_cap_lens[i]].tolist()
                hypotheses.append(pred_idx)
            references.extend(all_captionss)

            # compute loss and acc
            loss, acc = net.compute_loss(predictions, alphas, captions, cap_lens)
            losses.update(loss.item())
            accs.update(acc.item())
    

    assert len(references) == len(hypotheses)
    blue4 = corpus_bleu(references, hypotheses)

    # # For debug
    # with open('./temp_valid_result.txt', 'w') as fp:
    #     for h in hypotheses:
    #         h = [idx2word[i] for i in h]
    #         fp.writelines(' '.join(h) + '\n')
    
    # with open('./temp_valid_result_ref.txt', 'w') as fp:
    #     for r in references[0]:
    #         r = [idx2word[i] for i in r]
    #         fp.writelines(' '.join(r) + '\n')

    print('[INFO] BLEU-4 score: {:.4f} | Avg loss: {:.4f} | Avg acc: {:.4f}'.format(blue4, losses.avg, accs.avg))
    return blue4


# # debug
# word_map = json.load(open('./data_config/DefaultWordMap.json'))
# idx2word = {v: k for k, v in word_map.items()}

def evaluate(net: nn.Module, dataloader: DataLoader, device: torch.device, beam_searcher: models.BeamSearcher):
    references = []
    hypotheses = []

    print('[INFO] Start evaluating...')

    net.eval()
    with torch.no_grad():
        for i, (image_ids, images, captions, cap_lens, all_captionss) in enumerate(tqdm(dataloader)):
            # image_ids: tuple, shape=(batch_size,)
            # images: shape=(batch_size, 3, H, W)
            # captions: word indexes, shape=(batch_size, cap_len)
            # cap_lens: true caption length, shape=(batch_size,)
            # all_captionss: tuple of list of list, shape=(batch_size, n_catrgories, cap_len)

            # Move to GPU if available
            images = images.to(device)
            captions, scores, attention_weights = net.predict(images, beam_searcher, return_best=True)

            hypotheses.extend(captions) # filter out the <bos>
            references.extend(all_captionss)
    
    assert len(references) == len(hypotheses)
    blue4 = corpus_bleu(references, hypotheses)

    # For debug
    # with open('./temp_valid_result.txt', 'w') as fp:
    #     for h in hypotheses:
    #         h = [idx2word[i] for i in h]
    #         fp.writelines(' '.join(h) + '\n\n')
    # with open('./temp_valid_result.txt', 'w') as fp:
    #     # for hs in hypotheses:
    #     for h in hypotheses:
    #         h = [idx2word[i] for i in h]
    #         fp.writelines(' '.join(h) + '\n')
    #         # fp.writelines('\n')
    
    # with open('./temp_valid_result_ref.txt', 'w') as fp:
    #     for rs in references:
    #         for r in rs:
    #             r = [idx2word[i] for i in r]
    #             fp.writelines(' '.join(r) + '\n')
    #         fp.writelines('\n')

    print('[INFO] Beam size: {} | BLEU-4 score: {:.4f}'.format(beam_searcher.beam_size, blue4))


if __name__ == "__main__":
    # dataset and dataloader config
    dataset_guide_path = './data_config/ValDatasetGuide.json'
    word_map_path      = './data_config/SelectGloveWordMap.json'

    # net checkpoint
    checkpoint         = './checkpoints/checkpoint_best.pth'

    device      = torch.device("cuda:0")
    num_workers = 4
    batch_size  = 16
    pin_memory  = False

    # Build dataset and dataloader
    dataset = CaptionDataset(dataset_guide_path, word_map_path, remove_invalid=True, test_only_n_samples=None)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=dataset.collate_fn, pin_memory=pin_memory)

    checkpoint = torch.load(checkpoint)
    net = checkpoint['model']

    with open(word_map_path, 'r') as j:
        word_map = json.load(j)
    beam_searcher = models.BeamSearcher(word_map['<bos>'], word_map['<eos>'], beam_size=5, max_steps=100, device=device)

    net = net.to(device)
    evaluate(net, dataloader, device, beam_searcher)
    
