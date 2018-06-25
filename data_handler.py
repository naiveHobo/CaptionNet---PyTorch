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
        self.wordmap = {}
        self.embeddings = None
        self.embed_size = 300
        self.PAD = '<pad>'
        self.UNKNOWN = '<unk>'
        self.START = '<start>'
        self.END = '<end>'
        self.augment = None

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
        self.wordmap = {0: self.PAD, 1: self.UNKNOWN, 2: self.START, 3: self.END}

        with tqdm(total=self.data.shape[0]) as pbar:
            for i, row in self.data.iterrows():
                caption = row['caption'].decode('utf').lower()
                caption = re.sub(r'[^\w\s\']', '', caption)

                tokens = caption.split(' ')
                tokens = tokens[:self.max_length]

                for token in tokens:
                    if token not in self.vocab:
                        self.vocab[token] = len(self.vocab)
                        self.wordmap[self.vocab[token]] = token

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

    def resize_images(self, out_path, img_size):
        """Resizes images in the dataset"""
        assert self.data is not None, "No data has been loaded yet, call DataHandler.read()"

        if not os.path.exists(out_path):
            os.makedirs(out_path)

        self.img_size = img_size

        print "\nResizing images..."

        with tqdm(total=self.data.shape[0]) as pbar:
            for i, row in self.data.iterrows():
                path = row['filename']
                img_name = path.split('/')[-1]
                image = Image.open(path).convert('RGB')
                image = image.resize([self.img_size, self.img_size], Image.ANTIALIAS)
                image.save(os.path.join(out_path, img_name), image.format)
                self.data.at[i, 'filename'] = os.path.join(out_path, img_name)

                pbar.update(1)

    def load_embeddings(self, embed_path, embed_size=300):
        """Loads pre-trained embeddings"""
        assert self.vocab, "Vocabulary has not been built yet, call DataHandler.build_vocab()"

        print "\nLoading pre-trained embeddings..."

        embed = []
        vocab = {self.PAD: 0, self.UNKNOWN: 1, self.START: 2, self.END: 3}
        wordmap = {0: self.PAD, 1: self.UNKNOWN, 2: self.START, 3: self.END}

        embed.append(np.zeros(embed_size, dtype=np.float32))
        embed.append(np.random.uniform(-0.1, 0.1, embed_size))
        embed.append(np.random.uniform(-0.1, 0.1, embed_size))
        embed.append(np.random.uniform(-0.1, 0.1, embed_size))

        with gzip.open(embed_path, 'r') as fEmbeddings:
            for i, line in tqdm(enumerate(fEmbeddings)):
                split = line.decode('utf-8').strip().split(' ')
                word = split[0].lower()

                if word in self.vocab:
                    vector = np.array([float(num) for num in split[1:]], dtype=np.float32)
                    embed.append(vector)
                    vocab[word] = len(vocab)
                    wordmap[vocab[word]] = word

        self.vocab_size = len(vocab)
        self.vocab = vocab
        self.wordmap = wordmap
        self.embeddings = np.array(embed, dtype=np.float32)

        print "\nSuccessfully loaded pre-trained GloVe embeddings!"

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
                'wordmap': self.wordmap,
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
        self.wordmap = data['wordmap']

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
