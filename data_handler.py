import torch
import torch.utils.data as data
import numpy as np
from tqdm import tqdm
from nltk.tokenize import word_tokenize
from PIL import Image
import os
import cPickle as pkl
import gzip
import re


class DataHandler(data.Dataset):
    def __init__(self, data=None, max_len=20):
        self.data = data
        self.max_length = max_len
        self.img_size = 224
        self.vocab_size = 0
        self.vocab = {}
        self.word_id = {}
        self.embeddings = None
        self.embed_size = 300
        self.PAD = '<pad>'
        self.UNKNOWN = '<unk>'
        self.START = '<start>'
        self.END = '<end>'
        self.augment = None
        self.mean = None #np.array([0.45803204, 0.4461004, 0.4039198])
        self.std = None #np.array([0.24218813, 0.23319727, 0.23719482])

    def __getitem__(self, index):
        """Returns an image and caption pair from the dataset"""
        path = self.data['filename'].iloc[index]
        caption = self.data['caption'].iloc[index]

        image = Image.open(path).convert('RGB')
        if self.augment is not None:
            image = self.augment(image)

        caption = caption.decode('utf').lower().strip()
        caption = re.sub(r'[^\w\s\<\>]', '', caption)
        tokens = caption.split(' ')
        caption = [self.get_word_id(token) for token in tokens]
        target = torch.LongTensor(caption)

        return image, target

    def read(self, data, max_len=20):
        """Read DataFrame"""
        self.data = data
        self.max_length = max_len

    def build_vocab(self):
        """Cleans the data and creates the vocabulary"""
        assert self.data is not None, "No data has been loaded yet, call DataHandler.read()"

        print "\nBuilding vocabulary..."

        self.vocab = {self.PAD: 0, self.UNKNOWN: 1, self.START: 2, self.END: 3}
        self.word_id = {0: self.PAD, 1: self.UNKNOWN, 2: self.START, 3: self.END}

        with tqdm(total=self.data.shape[0]) as pbar:
            for i, row in self.data.iterrows():
                caption = row['caption'].decode('utf').lower()
                caption = re.sub(r'[^\w\s\']', '', caption)

                tokens = word_tokenize(caption)
                tokens = tokens[:self.max_length]

                for token in tokens:
                    if token not in self.vocab:
                        self.vocab[token] = len(self.vocab)
                        self.word_id[self.vocab[token]] = token

                caption = " ".join(tokens)
                self.data.at[i, 'caption'] = caption

                pbar.update(1)

        self.vocab_size = len(self.vocab)

        print "\nVocabulary was successfully built!"

    def pad_data(self):
        """Pads data sequences to the max length"""
        assert self.data is not None, "No data has been loaded yet, call DataHandler.read()"

        print "\nStarting the padding process..."

        with tqdm(total=self.data.shape[0]) as pbar:
            for i, row in self.data.iterrows():
                caption = row['caption'].decode('utf').lower()
                caption = re.sub(r'[^\w\s\']', '', caption)
                tokens = word_tokenize(caption)
                tokens = tokens[:self.max_length]
                tokens = [self.START] + tokens + [self.END]
                self.max_length += 2

                for _ in range(self.max_length - len(tokens)):
                    tokens.append(self.PAD)

                caption = " ".join(tokens)

                self.data.at[i, 'caption'] = caption

                pbar.update(1)

        print "\nPadding was successful!"

    def resize_images(self, out_path):
        """Resizes images in the dataset and calculates mean and standard deviation"""
        assert self.data is not None, "No data has been loaded yet, call DataHandler.read()"

        if not os.path.exists(out_path):
            os.makedirs(out_path)

        if self.mean is None and self.std is None:
            self.mean = np.zeros(3)
            self.std = np.zeros(3)
            calc_mean = True
        else:
            calc_mean = False

        print "\nResizing images and calculating mean and standard deviation..."

        with tqdm(total=self.data.shape[0]) as pbar:
            for i, row in self.data.iterrows():
                path = row['filename']
                img_name = path.split('/')[-1]
                image = Image.open(path).convert('RGB')
                image = image.resize([self.img_size, self.img_size], Image.ANTIALIAS)
                image.save(os.path.join(out_path, img_name), image.format)
                self.data.at[i, 'filename'] = os.path.join(out_path, img_name)

                if calc_mean:
                    image = np.array(image)
                    self.mean += np.mean(image, axis=(0, 1))
                    self.std += np.std(image, axis=(0, 1))

                pbar.update(1)

        if calc_mean:
            self.mean /= self.data.shape[0] * 255
            self.std /= self.data.shape[0] * 255

    def load_embeddings(self, embed_path, embed_size=300):
        """Loads pre-trained embeddings"""
        assert self.vocab, "Vocabulary has not been built yet, call DataHandler.build_vocab()"

        print "\nLoading pre-trained embeddings..."

        self.embeddings = np.random.uniform(-0.1, 0.1, [len(self.vocab), embed_size])
        self.embeddings[self.vocab[self.PAD]] = np.zeros(embed_size, dtype=np.float32)

        with gzip.open(embed_path, 'r') as fEmbeddings:
            for i, line in tqdm(enumerate(fEmbeddings)):
                split = line.decode('utf-8').strip().split(' ')
                word = split[0].lower()

                if word in self.vocab:
                    vector = np.array([float(num) for num in split[1:]], dtype=np.float32)
                    self.embeddings[self.vocab[word]] = vector

        print "\nSuccessfully loaded pre-trained embeddings!"

    def get_word_id(self, token):
        """Returns the id of a token"""
        if token in self.vocab:
            return self.vocab[token]
        elif token.lower() in self.vocab:
            return self.vocab[token.lower()]

        return self.vocab[self.UNKNOWN]

    def save_data(self, out_path='./data/processed.pkl.gz'):
        """Saves the embeddings and vocab as a zipped pickle file"""

        assert (self.embeddings is not None or self.vocab), "Data has not been processed yet"

        data = {'embeddings': self.embeddings,
                'vocab': self.vocab,
                'wordmap': self.word_id,
                'mean': self.mean,
                'std': self.std
                }

        with gzip.open(out_path, 'wb') as out_file:
            pkl.dump(data, out_file)

        print "\nData stored as {}".format(out_path)

    def load_data(self, path):
        """Loads embeddings and vocab from a zipped pickle file"""

        with gzip.open(path, 'rb') as in_file:
            data = pkl.load(in_file)

        self.embeddings = data['embeddings']
        self.embed_size = self.embeddings.shape[1]
        self.vocab = data['vocab']
        self.vocab_size = len(self.vocab)
        self.word_id = data['wordmap']
        self.mean = data['mean']
        self.std = data['std']

        print "\nSuccessfully loaded data from {}".format(path)

    def set_augment(self, transform):
        """Sets transform for dataset"""
        self.augment = transform

    def __len__(self):
        return self.data.shape[0]

    @staticmethod
    def collate_fn(data):
        """Creates mini-batch tensors from the list of tuples (image, caption)."""
        # Sort a data list by caption length (descending order).
        data.sort(key=lambda x: len(x[1]), reverse=True)
        images, captions = zip(*data)

        # Merge images (from tuple of 3D tensor to 4D tensor).
        images = torch.stack(images, 0)

        # Merge captions (from tuple of 1D tensor to 2D tensor).
        lengths = [len(cap) for cap in captions]
        targets = torch.zeros(len(captions), max(lengths)).long()
        for i, cap in enumerate(captions):
            end = lengths[i]
            targets[i, :end] = cap[:end]
        return images, targets, lengths
