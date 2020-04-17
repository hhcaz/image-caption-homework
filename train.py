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


def train(net: nn.Module, dataloader: DataLoader, optimizer: optim.Optimizer, device: torch.device, epoch, print_freq=100):
    total_batches = len(dataloader)

    data_load_time   = AverageMeter()
    batch_total_time = AverageMeter()
    losses           = AverageMeter()
    accs             = AverageMeter()

    start_time_stamp = time.time()

    net.train()
    for i, (image_ids, images, captions, cap_lens, all_captionss) in enumerate(dataloader):
        # image_ids: tuple, shape=(batch_size,)
        # images: shape=(batch_size, 3, H, W)
        # captions: word indexes, shape=(batch_size, cap_len)
        # cap_lens: true caption length, shape=(batch_size,)
        # all_captionss: tuple of list of list, shape=(batch_size, n_catrgories, cap_len)
        data_load_time.update(time.time() - start_time_stamp)

        # Move to GPU if available
        images   = images.to(device)
        captions = captions.to(device)
        cap_lens = cap_lens.to(device)

        # Forward prop + backward prop
        optimizer.zero_grad()
        # predictions' shape=(batch_size, max_cap_len, vocab_size)
        # alphas' shape=(batch_size, num_pixels)
        # predictions usual end with <eos> but <bos> is excluded
        # compute loss and acc
        predictions, alphas = net(images, captions, cap_lens)
        loss, acc = net.compute_loss(predictions, alphas, captions, cap_lens)

        loss.backward()
        optimizer.step()

        # Update recorder
        batch_total_time.update(time.time() - start_time_stamp)
        losses.update(loss.item())
        accs.update(acc.item())

        start_time_stamp = time.time()

        if i % print_freq == 0:
            print('[INFO] Epoch: {0} | Batches: {1}/{2} | '
                'Data load time: {data_load_time.current:.3f} ({data_load_time.avg:.3f}) | '
                'Batch time: {batch_total_time.current:.3f} ({batch_total_time.avg:.3f}) | '
                'Loss: {losses.current:.4f} ({losses.avg:.4f}) | '
                'Acc: {accs.current:.4f} ({accs.avg:.4f})'
                .format(epoch, i, total_batches,
                    data_load_time=data_load_time,
                    batch_total_time=batch_total_time,
                    losses=losses,
                    accs=accs))

# ############### debug #########################
# word_map = json.load(open('./data_config/SelectGloveWordMap.json', 'r', encoding='utf-8'))
# idx2word = {v: k for k, v in word_map.items()}
# ################################################

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

    # ############# For debug ########################
    # with open('./temp_valid_result.txt', 'w') as fp:
    #     for h in hypotheses:
    #         h = [idx2word[i] for i in h]
    #         fp.writelines(' '.join(h) + '\n')
    
    # with open('./temp_valid_result_ref.txt', 'w') as fp:
    #     for r in references[0]:
    #         r = [idx2word[i] for i in r]
    #         fp.writelines(' '.join(r) + '\n')
    # ################################################

    print('[INFO] BLEU-4 score: {:.4f} | Avg loss: {:.4f} | Avg acc: {:.4f}'.format(blue4, losses.avg, accs.avg))
    return blue4


def save_checkpoint(epoch, net: nn.Module, optimizer: optim.Optimizer, bleu4, is_best, out_folder='./checkpoints'):
    state = {
        'epoch': epoch,
        'model': net,
        'optimizer': optimizer,
        'bleu4': bleu4,
    }

    if not os.path.exists(out_folder):
        os.mkdir(out_folder)

    file_name = 'checkpoint.pth'
    file_path = os.path.join(out_folder, file_name)
    torch.save(state, file_path)

    if is_best:
        file_name = 'checkpoint_best.pth'
        file_path = os.path.join(out_folder, file_name)
        torch.save(state, file_path)


def clip_gradient(optimizer: optim.Optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def adjust_learning_rate(optimizer, shrink_factor):
    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))



if __name__ == "__main__":
    # basic config
    device          = torch.device("cuda:0")
    cudnn.benchmark = True

    # train procedure
    epoches = 20
    lr_rate = 1e-3

    fine_tune_encoder = False

    # model config
    model_name    = 'ShowAttendTell'
    encoder_dim   = 2048
    decoder_dim   = 512
    attention_dim = 512
    embedding_dim = 300
    dropout       = 0.5
    checkpoint    = None #'./checkpoints/checkpoint_best.pth'

    # dataset and dataloader config
    train_dataset_guide_path = './data_config/TrainDatasetGuide.json'
    valid_dataset_guide_path = './data_config/ValDatasetGuide.json'
    word_map_path            = './data_config/SelectGloveWordMap.json'

    # whether to use glove
    glove_tensor_path        = './data_config/SelectGloveTensor.pth'
    use_glove                = True

    train_num_workers = 4
    valid_num_workers = 4
    train_batch_size  = 32
    valid_batch_size  = 32

    # Build dataset and dataloader
    train_dataset = CaptionDataset(train_dataset_guide_path, word_map_path, remove_invalid=True)
    valid_dataset = CaptionDataset(valid_dataset_guide_path, word_map_path, remove_invalid=True)
    vocab_size    = train_dataset.vocab_size

    train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=train_num_workers, collate_fn=train_dataset.collate_fn)
    valid_dataloader = DataLoader(valid_dataset, batch_size=valid_batch_size, shuffle=False, num_workers=valid_num_workers, collate_fn=valid_dataset.collate_fn)

    # ############################# Debug ########################################
    # # Build dataset and dataloader
    # train_dataset = CaptionDataset(train_dataset_guide_path, word_map_path, remove_invalid=True, test_only_n_samples=512)
    # valid_dataset = CaptionDataset(train_dataset_guide_path, word_map_path, remove_invalid=True, test_only_n_samples=512)
    # vocab_size    = train_dataset.vocab_size

    # train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=False, num_workers=train_num_workers, collate_fn=train_dataset.collate_fn)
    # valid_dataloader = DataLoader(valid_dataset, batch_size=valid_batch_size, shuffle=False, num_workers=valid_num_workers, collate_fn=valid_dataset.collate_fn)
    # ############################################################################

    # load checkpoint if checkpoint exists
    if checkpoint is None:
        start_epoch = 0
        net = models.get_model(model_name, encoder_dim=encoder_dim, decoder_dim=decoder_dim, \
            attention_dim=attention_dim, embedding_dim=embedding_dim, dropout=dropout, vocab_size=vocab_size)
        net.encoder.fine_tune(fine_tune_encoder)

        if use_glove:
            from models.utils import ComposedEmbedding
            embed_tensor = torch.load(glove_tensor_path)
            trainable_size = vocab_size - embed_tensor.size(0)
            embeddings = ComposedEmbedding(embed_tensor.size(0), trainable_size, embedding_dim)
            embeddings.load_pretained(embed_tensor)
            # embeddings.fine_tune_all(True)
            net.decoder_with_attention.load_pretrained_embeddings(embeddings)

        optimizer = optim.Adam(params=filter(lambda p: p.requires_grad, net.parameters()), lr=lr_rate)
        best_bleu4 = 0
    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        net = checkpoint['model']
        optimizer = checkpoint['optimizer']
        best_bleu4 = checkpoint['bleu4']


    net = net.to(device)
    for epoch in range(start_epoch, start_epoch + epoches):
        train(net=net, dataloader=train_dataloader, optimizer=optimizer, device=device, epoch=epoch, print_freq=2)
        current_bleu4 = validate(net=net, dataloader=valid_dataloader, device=device)
        
        is_best = current_bleu4 > best_bleu4
        best_bleu4 = max(current_bleu4, best_bleu4)

        save_checkpoint(epoch, net=net, optimizer=optimizer, bleu4=current_bleu4, is_best=is_best, out_folder='./checkpoints')
    
