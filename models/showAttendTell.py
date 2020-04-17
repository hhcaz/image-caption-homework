import torch
import torch.nn as nn
import torchvision
from torch.nn.utils.rnn import pack_padded_sequence
from .utils import BeamSearcher

class ImageEncoder(nn.Module):
    def __init__(self, encoded_wh=14):
        super(ImageEncoder, self).__init__()
        self.encoded_wh = encoded_wh
        
        resnet = torchvision.models.resnet101(pretrained=False)
        net_weights = torch.load('./models/resnet101-5d3b4d8f.pth')
        resnet.load_state_dict(net_weights)

        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        self.avgpool = nn.AdaptiveAvgPool2d((self.encoded_wh, self.encoded_wh))

        self.fine_tune(False)
    
    def fine_tune(self, yes=True):
        for p in self.resnet.parameters():
            p.requires_grad = False
        
        for child in list(self.resnet.children())[5:]:
            for p in child.parameters():
                p.requires_grad = yes
    
    def forward(self, x):
        x = self.resnet(x) # (batch_size, 2048, image_h/32, image_w/32)
        x = self.avgpool(x) # (batch_size, 2048, encoded_wh, encoded_wh)
        x = x.permute(0, 2, 3, 1) # (batch_size, encoded_wh, encoded_wh, 2048)

        return x


class Attention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)  # linear layer to transform encoded image
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)  # linear layer to transform decoder's output
        self.full_att = nn.Linear(attention_dim, 1)  # linear layer to calculate values to be softmax-ed
        self.relu = nn.ReLU(inplace=True)
        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights

    def forward(self, encoder_out, decoder_hidden):
        """
        Arguments:
        - encoder_out: encoded images, shape=(batch_size, num_pixels, encoder_dim)
        - decoder_hidden: previous decoder output, shape=(batch_size, decoder_dim)
        
        Returns:
        - attention_weighted_encoding: shape=(batch_size, encoder_dim)
        - alpha: attention weights, shape=(batch_size, num_pixels)
        """
        att1 = self.encoder_att(encoder_out)  # (batch_size, num_pixels, attention_dim)
        att2 = self.decoder_att(decoder_hidden)  # (batch_size, attention_dim)
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)  # (batch_size, num_pixels)
        alpha = self.softmax(att)  # (batch_size, num_pixels)
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # (batch_size, encoder_dim)

        return attention_weighted_encoding, alpha


class DecoderWithAttention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim, vocab_size, embedding_dim, dropout=0.5):
        super(DecoderWithAttention, self).__init__()
        self.encoder_dim   = encoder_dim
        self.decoder_dim   = decoder_dim
        self.attention_dim = attention_dim
        self.vocab_size    = vocab_size
        self.embedding_dim = embedding_dim
        self.dropout       = dropout

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)
        self.dropout   = nn.Dropout(p=dropout)
        self.decoder   = nn.LSTMCell(embedding_dim + encoder_dim, decoder_dim, bias=True)
        self.init_h    = nn.Linear(encoder_dim, decoder_dim)
        self.init_c    = nn.Linear(encoder_dim, decoder_dim)
        self.f_beta    = nn.Linear(decoder_dim, encoder_dim)
        self.sigmoid   = nn.Sigmoid()
        self.fc        = nn.Linear(decoder_dim, vocab_size)

        self.init_weights()
    
    def init_weights(self):
        self.embedding.weight.detach().uniform_(-0.1, 0.1)
        self.fc.weight.detach().uniform_(-0.1, 0.1)
        self.fc.bias.detach().fill_(0)
    
    def load_pretrained_embeddings(self, embeddings):
        if isinstance(embeddings, torch.Tensor):
            self.embedding.weight = nn.Parameter(embeddings, requires_grad=False)
        elif isinstance(embeddings, nn.Module):
            self.embedding = embeddings
    
    def init_hidden_state(self, encoder_out):
        encoder_out = encoder_out.mean(dim=1) # shape=(batch_size, encoder_dim)
        h = self.init_h(encoder_out) # shape=(batch_size, decoder_dim)
        c = self.init_c(encoder_out) # shape=(batch_size, decoder_dim)

        return h, c
    
    def forward(self, encoder_out, captions, cap_lens):
        """
        Arguments:
        - encoder_out: encoded images, shape=(batch_size, h, w, encoder_dim)
        - captions: word indexes (include 'bos', 'eos' and 'pad'), shape=(batch_size, max_cap_len)
        - cap_lens: true captions' lengths, shape=(batch_size,)
            Note: the inputs are already sorted by cap_lens in CaptionDataset's collate_fn
        
        Returns:
        - predictions: raw predictions (not softmaxed), shape=(batch_size, max_cap_len, vocab_size)
            predictions usual end with <eos> but <bos> is excluded
        - alphas: attention weights on image pixels, shape=(batch_size, max_cap_len, num_pixels)
        """
        batch_size, encoder_h, encoder_w, encoder_dim = encoder_out.size()
        device = encoder_out.device

        # Flatten image
        num_pixels = encoder_h * encoder_w
        encoder_out = encoder_out.view(batch_size, num_pixels, encoder_dim)

        # Embedding
        embeddings = self.embedding(captions) # shape=(batch_size, L, embedding_dim)

        # Initialize LSTM state
        h, c = self.init_hidden_state(encoder_out) # shape=(batch_size, decoder_dim)

        # We don't decode at the <eos> position, since we've finished generating as soon as we generate <eos>
        # So, decoding lengths are actual lengths - 1
        decoder_cap_lens = cap_lens - 1 # decoder_cap_lens[0] >= decoder_cap_lens[1] >= ... >= decoder_cap_lens[-1]
        max_cap_len = decoder_cap_lens[0].item()

        # Create tensors to hold word prediction scores and alphas for attention
        predictions = torch.zeros(batch_size, max_cap_len, self.vocab_size).to(device)
        alphas = torch.zeros(batch_size, max_cap_len, num_pixels).to(device)

        # At each time-step, decode by
        # attention-weighing the encoder's output based on the decoder's previous hidden state output
        # then generate a new word in the decoder with the previous word and the attention weighted encoding
        for t in range(max_cap_len):
            batch_size_t = torch.sum(decoder_cap_lens > t) # captions whose length <= t will be ignored
            attention_weighted_encoding, alpha = self.attention(encoder_out[:batch_size_t], h[:batch_size_t])
            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))  # gating scalar, (batch_size_t, encoder_dim)
            attention_weighted_encoding = gate * attention_weighted_encoding # (batch_size_t, encoder_dim)

            h, c = self.decoder(
                torch.cat([embeddings[:batch_size_t, t, :], attention_weighted_encoding], dim=1),
                (h[:batch_size_t], c[:batch_size_t])
            ) # shape=(batch_size_t, decoder_dim)
            
            predictions[:batch_size_t, t, :] = self.fc(self.dropout(h)) # shape=(batch_size_t, vocab_size)
            alphas[:batch_size_t, t, :] = alpha # shape=(batch_size_t, num_pixels)
        
        return predictions, alphas
    

    def predict(self, encoder_out, beam_searcher: BeamSearcher, return_best=True):
        """
        Arguments:
        - encoder_out: encoded image, shape=(1, h, w, encoder_dim)
        - beam_searcher: BeamSearcher object searching captions

        Returns:
        - caption: list, word indexes (include <bos> and <eos> but no <pad>) shape=(cap_len,)
        - score: numeric, score of the caption
        - attention_weights: tensor, (cap_len, num_pixels)
        """
        _, encoder_h, encoder_w, encoder_dim = encoder_out.size()
        num_pixels = encoder_h * encoder_w
        # Flatten the height and width
        encoder_out = encoder_out.view(1, num_pixels, encoder_dim)
        # We'll treat the problem as having a batch size of beam_size
        encoder_out = encoder_out.expand(beam_searcher.beam_size, num_pixels, encoder_dim) # (beam_size, num_pixels, encoder_dim)
        h, c = self.init_hidden_state(encoder_out) # shape=(beam_size, decoder_dim)

        beam_searcher.reset()
        beam_searcher.register_history_tracer('attention_weight')

        # next_word_idx stores top beam_size_t previous words at each step; now they're just index of <bos>
        next_word_idx = beam_searcher.start() # shape=(beam_size,)

        while not beam_searcher.is_terminated:
            embedding = self.embedding(next_word_idx) # (beam_size_t, embed_dim)
            attention_weighted_encoding, alpha = self.attention(encoder_out, h)
            # awe's shape=(beam_size_t, encoder_dim), alpha's shape=(beam_size_t, num_pixels)

            gate = self.sigmoid(self.f_beta(h)) # gating scalar, (batch_size_t, encoder_dim)
            attention_weighted_encoding = gate * attention_weighted_encoding # (beam_size_t, encoder_dim)

            h, c = self.decoder(torch.cat([embedding, attention_weighted_encoding], dim=1), (h, c))
            prediction = self.fc(self.dropout(h))

            # Step once of the beam searcher
            prev_beam_idx, next_word_idx, incomplete_beam_idx = beam_searcher.step(prediction, attention_weight=alpha)
            
            # Select the incomplete branches and next word index
            h = h[prev_beam_idx[incomplete_beam_idx]] # shape=(beam_size_t, decoder_dim)
            c = c[prev_beam_idx[incomplete_beam_idx]] # shape=(beam_size_t, decoder_dim)
            encoder_out = encoder_out[prev_beam_idx[incomplete_beam_idx]]
            next_word_idx = next_word_idx[incomplete_beam_idx] # shape=(beam_size_t,)

        caption, score, registered_tracer_content = beam_searcher.fetch_results(return_best=return_best)
        attention_weight = registered_tracer_content['attention_weight']

        return caption, score, attention_weight


class ShowAttendTell(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim, vocab_size, embedding_dim, dropout=0.5):
        super(ShowAttendTell, self).__init__()
        self.encoder_dim   = encoder_dim
        self.decoder_dim   = decoder_dim
        self.attention_dim = attention_dim
        self.vocab_size    = vocab_size
        self.embedding_dim = embedding_dim
        self.dropout       = dropout

        self.alpha_c = 1. # regularization parameter for 'doubly stochastic attention', as in the paper

        self.encoder = ImageEncoder(encoded_wh=14)
        self.decoder_with_attention = DecoderWithAttention(encoder_dim, decoder_dim, \
            attention_dim, vocab_size, embedding_dim, dropout)
        
        self.criterion = nn.CrossEntropyLoss()
        
    def forward(self, images, captions, cap_lens):
        """
        Arguments:
        - encoder_out: encoded images, shape=(batch_size, h, w, encoder_dim)
        - captions: word indexes (include 'bos', 'eos' and 'pad'), shape=(batch_size, max_cap_len)
        - cap_lens: true captions' lengths, shape=(batch_size,)
            Note: the inputs are already sorted by cap_lens in CaptionDataset's collate_fn
        
        Returns:
        - predictions: raw predictions (not softmaxed), shape=(batch_size, max_cap_len, vocab_size)
            predictions usual end with <eos> but <bos> is excluded
        - alphas: attention weights on image pixels, shape=(batch_size, num_pixels)
        """
        images = self.encoder(images)
        predictions, alphas = self.decoder_with_attention(images, captions, cap_lens)

        return predictions, alphas
    
    def compute_loss(self, predictions, alphas, captions, cap_lens):
        """
        Arguments:
        - predictions: raw scores (not softmaxed), shape=(batch_size, max_cap_len, vocab_size)
            Not include <bos> step but include <eos> step
        - alphas: attention weights on image pixels, shape=(batch_size, max_cap_len, num_pixels)
        - captions: word indexes, shape=(batch_size, max_cap_lens), include <bos> and <eos> step
        - cap_lens: true captions' lengths, shape=(batch_size,)

        Returns:
        - loss: loss, numeric
        - acc: accuracy, teacher forcing accuracy
        """
        # We decode with <bos>, so there is no <bos> in prediction (<eos> still exists in preditions)
        # We should remove the <bos> from targets (<eos> still exists in targets)
        targets = captions[:, 1:]
        decoder_cap_lens = cap_lens - 1

        predictions = pack_padded_sequence(predictions,  decoder_cap_lens, batch_first=True).data # (X, vocab_size)
        targets = pack_padded_sequence(targets, decoder_cap_lens, batch_first=True).data # (X,); X=sum(decoder_cap_lens)
        
        # Calculate loss
        loss = self.criterion(predictions, targets)
        # Add doubly stochastic attention regularization
        loss += self.alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

        # Calculate teacher forcing accuracy
        pred_idx = predictions.argmax(dim=-1)
        correct_mask = pred_idx == targets
        acc = torch.sum(correct_mask).float() / correct_mask.numel()

        return loss, acc
    
    # def compute_loss(self, images, captions, cap_lens):
    #     """
    #     Arguments:
    #     - images: shape=(batch_size, 3, 256, 256)
    #     - captions: word indexes, shape=(batch_size, max_cap_lens)
    #     - cap_lens: true captions' lengths, shape=(batch_size,)

    #     Returns:
    #     - loss: loss, numeric
    #     - acc: accuracy, teacher forcing accuracy
    #     """
    #     predictions, alphas = self(images, captions, cap_lens)
    #     # predictions' shape=(batch_size, max_cap_len, vocab_size)
    #     # alphas' shape=(batch_size, num_pixels)
        
    #     # We decode with <bos>, so there is no <bos> in prediction (<eos> still exists in preditions)
    #     # We should remove the <bos> from targets (<eos> still exists in targets)
    #     targets = captions[:, 1:]
    #     decoder_cap_lens = cap_lens - 1

    #     predictions = pack_padded_sequence(predictions,  decoder_cap_lens, batch_first=True).data # (X, vocab_size)
    #     targets = pack_padded_sequence(targets, decoder_cap_lens, batch_first=True).data # (X,); X=sum(cap_lens)
        
    #     # Calculate loss
    #     loss = self.criterion(predictions, targets)
    #     # Add doubly stochastic attention regularization
    #     loss += self.alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

    #     # Calculate teacher forcing accuracy
    #     pred_idx = predictions.argmax(dim=-1)
    #     correct_mask = pred_idx == targets
    #     acc = torch.sum(correct_mask).float() / correct_mask.numel()

    #     return loss, acc
    
    def predict(self, images, beam_searcher: BeamSearcher, return_best=True):
        """
        Arguments:
        - images: shape=(batch_size, 3, 256, 256)
        - beam_searcher: BeamSearcher Object

        Returns:
        - captions: list, shape=(batch_size, cap_len), caption in different batch may have different length
        - scores, list, shape=(batch_size,)
        - attention_weights: list, shape=(batch_size, cap_len, num_pixels)
        """
        captions = []
        scores = []
        attention_weights = []

        self.eval()
        with torch.no_grad():
            images = self.encoder(images)
            for image in images:
                image = image.unsqueeze(0)
                caption, score, attention_weight = self.decoder_with_attention.predict(image, beam_searcher, return_best=return_best)

                captions.append(caption)
                scores.append(score)
                attention_weights.append(attention_weight)
        
        return captions, scores, attention_weights
