import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import json
import os


class CaptionDataset(Dataset):
    def __init__(self, dataset_guide_path, word_map_path, remove_invalid=True, test_only_n_samples=None):
        with open(dataset_guide_path, 'r', encoding='utf-8') as j:
            self.dataset_guide = json.load(j)
        assert 'test' not in self.dataset_guide['description']['splits'], \
            "CaptionDataset cannot receive split test which has no captions."
        print('[INFO] CaptionDatset of split(s) {} built.'.format(self.dataset_guide['description']['splits']))

        with open(word_map_path, 'r', encoding='utf-8') as j:
            self.word_map = json.load(j)
        
        self.image_root = self.dataset_guide['description']['image_root']
        self.samples = self.dataset_guide['samples']
        if remove_invalid:
            num_invalid = len([1 for s in self.samples if not s['is_valid']])
            self.samples = [s for s in self.samples if s['is_valid']]
            print('[INFO] Remove total {} invalid (pre-canned or rejected) samples'.format(num_invalid))
        
        # Debug, test only few samples to see if all the things works well
        if test_only_n_samples is not None:
            if test_only_n_samples > 0:
                test_only_n_samples = min(test_only_n_samples, len(self.samples))
            self.samples = self.samples[:test_only_n_samples]
        
        # When in validation, we need a caption along with all captions belong to the same image
        # thus we build a dict, the key is image id while the value is a list containing all
        # the sample index belonging to this image
        self.image_id_to_smaple_idx = {}
        for i, sample in enumerate(self.samples):
            image_id = sample['image_id']
            if image_id in self.image_id_to_smaple_idx:
                self.image_id_to_smaple_idx[image_id].append(i)
            else:
                self.image_id_to_smaple_idx[image_id] = [i]

        self.transformer = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            # transforms.Normalize([], [])
        ])
    
    @property
    def vocab_size(self):
        return len(self.word_map)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Returns:
        - image_id: numeric
        - image: tensor, shape=(3, H, W)
        - caption: list of word indexes, shape=(cap_len,), also includes <bos> and <eos>
        - all_caption: list of list of word indexes, shape=(n_categories, cap_lens), only include original sentences and <eos>
        """
        sample = self.samples[idx]
        image_id = sample['image_id']
        image_path = os.path.join(self.image_root, os.path.join(sample['split'], sample['image_name']))
        image = self.transformer(Image.open(image_path))

        tokens = sample['tokens']
        # Add 'bos' and 'eos' tokens
        caption = [self.word_map['<bos>']] + [self.word_map.get(word, self.word_map['<unk>']) for word in tokens] + [self.word_map['<eos>']]

        # Returns all captions for validation
        all_captions = [self.samples[i]['tokens'] for i in self.image_id_to_smaple_idx[image_id]] # (n_categories, cap_lens)
        all_captions = [[self.word_map.get(word, self.word_map['<unk>']) for word in caption] + [self.word_map['<eos>']] for caption in all_captions]
        # We don't include the index of <bos> here since <bos> is no use when validatig while <eos> is still important when evaluating
        
        return image_id, image, caption, all_captions
    
    def collate_fn(self, batch):
        """
        Returns:
        - image_ids: tuple, shape=(batch_size,)
        - images: shape=(batch_size, 3, H, W)
        - captions: word indexes, shape=(batch_size, cap_len)
        - cap_lens: true caption length, shape=(batch_size,)
        - all_captionss: tuple of list of list, shape=(batch_size, n_catrgories, cap_len)
        """
        # Sort samples in batch according to caption's length
        batch.sort(key=lambda x: len(x[2]), reverse=True)
        image_ids, images, captions, all_captionss = zip(*batch)
        images = torch.stack(images, dim=0)

        cap_lens = [len(caption) for caption in captions]
        max_cap_len = cap_lens[0]
        for caption in captions[1:]:
            # Pad to align with the longest one
            caption += [self.word_map['<pad>']] * (max_cap_len - len(caption))
        
        captions = torch.as_tensor(captions).long()
        cap_lens = torch.as_tensor(cap_lens).long()

        return image_ids, images, captions, cap_lens, all_captionss


class TestDataset(Dataset):
    def __init__(self, dataset_guide_path):
        with open(dataset_guide_path, 'r', encoding='utf-8') as j:
            self.dataset_guide = json.load(j)
        
        if 'train' in self.dataset_guide['description']['splits']:
            print('[WARN] TestDataset receives split: train.')
        if 'val' in self.dataset_guide['description']['splits']:
            print('[WARN] TestDataset receives split: val.')
        
        self.image_root = self.dataset_guide['description']['image_root']
        self.samples = self.dataset_guide['samples']

        self.transformer = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            # transforms.Normalize([], [])
        ])
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        image_id = sample['image_id']
        image_path = os.path.join(self.image_root, os.path.join(sample['split'], sample['image_name']))
        image = self.transformer(Image.open(image_path))

        return image_id, image
    
    def collate_fn(self, batch):
        image_ids, images = zip(*batch)
        images = torch.stack(images, dim=0)

        return image_ids, images


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    import time

    num_workers = 4
    start_time = time.time()
    dataset = CaptionDataset('./data_config/TrainDatasetGuide.json', './data_config/DefaultWordMap.json')
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=num_workers, collate_fn=dataset.collate_fn)
    for i, (image_ids, images, captions, cap_lens, all_captionss) in enumerate(tqdm(dataloader)):
        pass

    end_time = time.time()
    print("Time cost: {} s".format(end_time-start_time))