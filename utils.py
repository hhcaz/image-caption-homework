import json
import os
from collections import Counter
import random
import numpy as np
from tqdm import tqdm
import sys
import subprocess
import tempfile
import itertools
import torch

# class PTBTokenizer:
#     """Python wrapper of Stanford PTBTokenizer"""
#     # path to the stanford corenlp jar
#     STANFORD_CORENLP_3_4_1_JAR = 'stanford-corenlp-3.4.1.jar'
#     # punctuations to be removed from the sentences
#     PUNCTUATIONS = ["''", "'", "``", "`", "-LRB-", "-RRB-", "-LCB-", "-RCB-", \
#             ".", "?", "!", ",", ":", "-", "--", "...", ";"] 

#     @classmethod
#     def tokenize(cls, captions_for_image):
#         cmd = ['java', '-cp', STANFORD_CORENLP_3_4_1_JAR, \
#                 'edu.stanford.nlp.process.PTBTokenizer', \
#                 '-preserveLines', '-lowerCase']

#         # ======================================================
#         # prepare data for PTB Tokenizer
#         # ======================================================
#         final_tokenized_captions_for_image = {}
#         image_id = [k for k, v in captions_for_image.items() for _ in range(len(v))]
#         sentences = '\n'.join([c['caption'].replace('\n', ' ') for k, v in captions_for_image.items() for c in v])

#         # ======================================================
#         # save sentences to temporary file
#         # ======================================================
#         path_to_jar_dirname = os.path.dirname(os.path.abspath(__file__))
#         tmp_file = tempfile.NamedTemporaryFile(delete=False, dir=path_to_jar_dirname)
#         tmp_file.write(sentences.encode(encoding='UTF-8'))
#         tmp_file.close()

#         # ======================================================
#         # tokenize sentence
#         # ======================================================
#         cmd.append(os.path.basename(tmp_file.name))
#         p_tokenizer = subprocess.Popen(cmd, cwd=path_to_jar_dirname, \
#                 stdout=subprocess.PIPE)
#         token_lines = p_tokenizer.communicate(input=sentences.rstrip())[0]
#         lines = token_lines.decode().split('\n')
#         # remove temp file
#         os.remove(tmp_file.name)

#         # ======================================================
#         # create dictionary for tokenized captions
#         # ======================================================
#         for k, line in zip(image_id, lines):
#             if not k in final_tokenized_captions_for_image:
#                 final_tokenized_captions_for_image[k] = []
#             tokenized_caption = ' '.join([w for w in line.rstrip().split(' ') \
#                     if w not in PUNCTUATIONS])
#             final_tokenized_captions_for_image[k].append(tokenized_caption)

#         return final_tokenized_captions_for_image

class PTBTokenizer:
    """Python wrapper of Stanford PTBTokenizer"""
    # path to the stanford corenlp jar
    STANFORD_CORENLP_3_4_1_JAR = 'stanford-corenlp-3.4.1.jar'
    # punctuations to be removed from the sentences
    PUNCTUATIONS = ["''", "'", "``", "`", "-LRB-", "-RRB-", "-LCB-", "-RCB-", \
            ".", "?", "!", ",", ":", "-", "--", "...", ";"]

    @classmethod
    def tokenize(cls, sentences):
        """
        Arguments:
        - sentences: list of sentence, example: ["A computer screen shows a repair prompt.", "A dog sits on the ground", ...]
        - tokenized_captions: list of list of tokens, example: 
            [['a', 'computer', 'screen', 'show', 'a', 'repair', 'prompt'], ['a', 'dog', 'sit', 'the', 'ground'], ...]
        """
        cmd = ['java', '-cp', cls.STANFORD_CORENLP_3_4_1_JAR, \
                'edu.stanford.nlp.process.PTBTokenizer', \
                '-preserveLines', '-lowerCase']

        # prepare data for PTB Tokenizer
        sentences = '\n'.join([s.replace('\n', ' ').replace('\r', ' ') for s in sentences])

        # save sentences to temporary file
        path_to_jar_dirname = os.path.dirname(os.path.abspath(__file__))
        tmp_file = tempfile.NamedTemporaryFile(delete=False, dir=path_to_jar_dirname)
        tmp_file.write(sentences.encode(encoding='UTF-8'))
        tmp_file.close()

        # tokenize sentence
        cmd.append(os.path.basename(tmp_file.name))
        p_tokenizer = subprocess.Popen(cmd, cwd=path_to_jar_dirname, stdout=subprocess.PIPE)
        token_lines = p_tokenizer.communicate(input=sentences.rstrip())[0]
        lines = token_lines.decode().split('\n')

        # remove temp file
        os.remove(tmp_file.name)

        tokenized_captions = []
        for line in lines:
            line = [w for w in line.rstrip().split(' ') if w not in cls.PUNCTUATIONS]
            tokenized_captions.append(line)

        return tokenized_captions


def create_dataset_guide(image_root, annotation_root, output_folder, splits=['train'], max_n_captions_per_image=5, max_len=100):
    """Generate a json file like:
    {
        "description": {
            "image_root": "path/to/image/root",
            "splits": ["val"],
            "min_word_freq": 5,
            "max_len": 100,
            "max_n_captions_per_image": 5,
            "n_total_images": 7750,
            "n_total_captions": 38750,
            "n_valid_images": 7542,
            "n_valid_captions": 33144
        },
        "samples": [
            {
                "tokens": ["a", "computer", "screen", "show", "a", "repair", "prompt", "on", "the", "screen"],
                "split": "val",
                "image_name": "VizWiz_val_00000000.jpg",
                "image_id": 23431,
                "is_valid": true
            },
            {
                "tokens": ["a", "computer", "screen", "with", "a", "repair", "automatically", "pop", "up"],
                "split": "val",
                "image_name": "VizWiz_val_00000000.jpg",
                "image_id": 23431,
                "is_valid": true
            },
            {
                "tokens": ["partial", "computer", "screen", "showing", "the", "need", "of", "repairs"],
                "split": "val",
                "image_name": "VizWiz_val_00000000.jpg",
                "image_id": 23431,
                "is_valid": true
            },
            ...
            ...
            ...
        ]
    }
    """
    splits = [split.lower() for split in splits]
    if 'test' in ' '.join(splits):
        assert len(splits) == 1, "Test split has no captions, cannot process together with other splits."

    description = {
        'image_root': image_root,
        'splits': splits,
        'max_n_captions_per_image': max_n_captions_per_image,
        'max_len': max_len,
        'n_total_images': 0,
        'n_total_captions': 0,
        'n_valid_images': 0,
        'n_valid_captions': 0
    }

    samples = []
    for split in splits:
        # Test split should be treat specially later
        if split.lower() == 'test':
            continue

        print('[INFO] Reading raw data from split: {}...'.format(split))
        json_path = os.path.join(annotation_root, split + '.json')
        with open(json_path, 'r') as j:
            data = json.load(j)

        images = data['images']
        annotations = data['annotations']
        for i, image in enumerate(tqdm(images)):
            image_name = image['file_name']
            image_id = image['id']
            image_path = os.path.join(image_root, os.path.join(split, image_name))
            if not os.path.exists(image_path):
                print('[INFO] Image {} not found, skipped.'.format(image_path))
                continue
            
            samples_per_image = []
            for j in range(5*i, 5*(i+1)):
                annotation = annotations[j]
                assert image['id'] == annotation['image_id'] # Sanity check

                # Mark annotaions which are pre-canned or rejected or length exceeds the limit
                is_valid = not (annotation['is_precanned'] or annotation['is_rejected'] or len(annotation['caption']) > max_len)
                samples_per_image.append({
                    'tokens': annotation['caption'], # now they are just the complete sentence, the batch tokenization will be done later
                    'split': split,
                    'image_name': image_name,
                    'image_id': image_id,
                    'is_valid': is_valid
                })
            
            # If num of samples for this image exceeds the limit, we just sample
            if len(samples_per_image) > max_n_captions_per_image:
                samples_per_image = random.sample(samples_per_image, k = max_n_captions_per_image)
            
            samples.extend(samples_per_image)
            num_valid_samples_this_image = len([1 for s in samples_per_image if s['is_valid']])
            description['n_valid_images'] += 1 if num_valid_samples_this_image > 0 else 0
            description['n_valid_captions'] += num_valid_samples_this_image
            
        description['n_total_images'] += len(images)
        description['n_total_captions'] += len(annotations)
    
    # Batch tokenize
    if len(samples) > 0:
        sentences = [sample['tokens'] for sample in samples]
        tokenized_captions = PTBTokenizer.tokenize(sentences)
        assert len(tokenized_captions) == len(samples)
        for i, caption in enumerate(tokenized_captions):
            samples[i]['tokens'] = caption
    
    # We then process the test split
    if 'test' in ' '.join(splits):
        split = 'test'
        print('[INFO] Reading raw data from split: {}...'.format(split))
        json_path = os.path.join(annotation_root, split + '.json')
        with open(json_path, 'r') as j:
            data = json.load(j)

        images = data['images']
        for i, image in enumerate(tqdm(images)):
            image_name = image['file_name']
            image_id = image['id']
            image_path = os.path.join(image_root, os.path.join(split, image_name))
            if not os.path.exists(image_path):
                print('[INFO] Image {} not found, skipped.'.format(image_path))
                continue
            samples.append({
                'split': split,
                'image_name': image_name,
                'image_id': image_id
            })

        description['n_total_images'] += len(images)
        description['n_valid_images'] += len(images)

    # Compose together
    dataset_guide = {'description': description, 'samples': samples}

    # Save dataset guide
    output_file_name = ''.join(map(str.capitalize, splits)) + 'DatasetGuide.json'
    output_path = os.path.join(output_folder, output_file_name)
    print('[INFO] Saving dataset guide to {}...'.format(output_path))
    with open(output_path, 'w') as j:
        json.dump(dataset_guide, j, indent=4)
    
    print('[INFO] Done. List dataset guide info:')
    print(dataset_guide['description'])

    return output_path


def create_defualt_word_map(dataset_guide_path, output_folder, min_word_freq=5):
    with open(dataset_guide_path, 'r') as j:
        dataset_guide = json.load(j)
    
    assert 'test' not in dataset_guide['description']['splits'], "Test split has no captions."
    
    word_freq = Counter()
    for sample in dataset_guide['samples']:
        tokens = sample['tokens']
        word_freq.update(tokens)
    
    words = [word for word in word_freq.keys() if word_freq[word] > min_word_freq]
    word_map = {word: index for index, word in enumerate(words)}
    word_map['<pad>'] = len(word_map)
    word_map['<unk>'] = len(word_map)
    word_map['<bos>'] = len(word_map)
    word_map['<eos>'] = len(word_map)

    output_path = os.path.join(output_folder, 'DefaultWordMap.json')
    with open(output_path, 'w') as j:
        json.dump(word_map, j, indent=4)
    
    return output_path


def create_word_map_from_Glove(glove_path, output_folder):
    word_map = {}
    with open(glove_path, 'r', encoding='utf-8') as fp:
        index = 0
        while True:
            line = fp.readline()
            if line.strip():
                token = line.split(' ')[0]
                word_map[token] = index
            else:
                break
            index += 1
    word_map['<pad>'] = len(word_map)
    word_map['<unk>'] = len(word_map)
    word_map['<bos>'] = len(word_map)
    word_map['<eos>'] = len(word_map)

    output_path = os.path.join(output_folder, 'GloveWordMap.json')
    with open(output_path, 'w', encoding='utf-8') as j:
        json.dump(word_map, j, indent=4, ensure_ascii=False)
    
    return output_path


def glove_to_tensor(glove_path, output_folder):
    embeddings = []
    with open(glove_path, 'r', encoding='utf-8') as fp:
        while True:
            line = fp.readline()
            if line.strip():
                embedding = [float(n) for n in line.split()[1:]]
                embeddings.append(embedding)
            else:
                break
    
    embeddings = torch.as_tensor(embeddings).float()
    output_path = os.path.join(output_folder, 'GloveTensor.pth')
    torch.save(embeddings, output_path)



def filter_glove_tensor_with_wanted_word_map(glove_tensor_path, glove_word_map_path, wanted_word_map_path, output_folder):
    glove_tensor = torch.load(glove_tensor_path)
    with open(wanted_word_map_path, 'r', encoding='utf-8') as j:
        wanted_word_map = json.load(j)
    with open(glove_word_map_path, 'r', encoding='utf-8') as j:
        glove_word_map = json.load(j)
    
    glove_word_map.pop('<pad>', None)
    glove_word_map.pop('<unk>', None)
    glove_word_map.pop('<bos>', None)
    glove_word_map.pop('<eos>', None)

    reverse_wanted_word_map = {v: k for k, v in wanted_word_map.items()}

    select_idx = []
    not_find_words = []
    new_word_map = {}

    new_idx = 0
    for idx in sorted(reverse_wanted_word_map):
        word = reverse_wanted_word_map[idx]
        if word not in glove_word_map:
            not_find_words.append(word)
            print('not find', word)
            continue
        
        select_idx.append(glove_word_map[word])
        new_word_map[word] = new_idx
        new_idx += 1
    
    for i, word in enumerate(not_find_words):
        new_word_map[word] = i + new_idx
    
    select_idx = torch.LongTensor(select_idx)
    wanted_tensor = glove_tensor[select_idx]

    tensor_path = os.path.join(output_folder, 'SelectGloveTensor.pth')
    torch.save(wanted_tensor, tensor_path)

    new_word_map_path = os.path.join(output_folder, 'SelectGloveWordMap.json')
    with open(new_word_map_path, 'w', encoding='utf-8') as j:
        json.dump(new_word_map, j, indent=4, ensure_ascii=False)

    print('total {} words not found'.format(len(not_find_words)))


class AverageMeter(object):
    def __init__(self, beta=0.9):
        self.beta = beta
        self.reset()
    
    def reset(self):
        self.current = 0
        self.count = 0
        self.avg = 0
        self.exp_avg = 0
        self.exp_avg_biased = 0
    
    def update(self, value):
        self.current = value
        self.avg = (self.count * self.avg + self.current) / (self.count + 1)
        self.count += 1
        self.exp_avg = self.beta * self.exp_avg + (1 - self.beta) * self.current
        self.exp_avg_biased = self.exp_avg / (1 - self.beta**self.count)



if __name__ == "__main__":
    image_root      = 'D:\\hhcaz\\ZJU\\Junior_spring_summer\\Machine Learning\\datasets\\VizWiz_Captions'
    annotation_root = 'D:\\hhcaz\\ZJU\\Junior_spring_summer\\Machine Learning\\datasets\\VizWiz_Captions\\annotations'
    output_folder   = '.\\data_config'

    # train_data_config = create_dataset_guide(image_root, annotation_root, output_folder, splits=['train'])
    # word_map_path = create_defualt_word_map(train_data_config, output_folder)

    # valid_data_config = create_dataset_guide(image_root, annotation_root, output_folder, splits=['val'])
    # test_data_config = create_dataset_guide(image_root, annotation_root, output_folder, splits=['test'])
    # glove_path = './xunlei/glove.6B.zip/glove.6B.300d.txt'
    # create_word_map_from_Glove(glove_path, output_folder)
    # glove_to_tensor(glove_path, output_folder)

    glove_tensor_path = './data_config/GloveTensor.pth'
    glove_word_map_path = './data_config/GloveWordMap.json'
    wanted_word_map_path = './data_config/DefaultWordMap.json'
    filter_glove_tensor_with_wanted_word_map(glove_tensor_path, glove_word_map_path, wanted_word_map_path, output_folder)
