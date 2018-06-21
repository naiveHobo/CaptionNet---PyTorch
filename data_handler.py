import numpy as np
from tqdm import tqdm
from nltk.tokenize import word_tokenize
from PIL import Image
import os
import cPickle as pkl
import gzip


class DataHandler:

    def __init__(self, data=None, max_len=20, augment=False):
        self.data = data
        self.max_length = max_len
        self.vocab = {}
        self.word_id = {}
        self.embeddings = None
        self.embed_size = 300
        self.PAD = '<pad>'
        self.UNKNOWN = '<unk>'
        self.START = '<start>'
        self.END = '<end>'
        self.augment = augment
        self.mean = None

    # def __getitem__(self, index):
    #     """Returns an image and caption pair from the dataset"""
    #
    #     path = self.data['filename'].iloc[index]
    #     caption = self.data['caption'].iloc[index]
    #
    #     image = Image.open(path).convert('RGB')
    #     if self.augment is not None:
    #         image = self.transform(image)
    #
    #     # Convert caption (string) to word ids.
    #     tokens = nltk.tokenize.word_tokenize(str(caption).lower())
    #     caption = []
    #     caption.append(vocab('<start>'))
    #     caption.extend([vocab(token) for token in tokens])
    #     caption.append(vocab('<end>'))
    #     target = torch.Tensor(caption)
    #     return image, target

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

                tokens = word_tokenize(caption)
                tokens = tokens[:self.max_length]

                for token in tokens:
                    if token not in self.vocab:
                        self.vocab[token] = len(self.vocab)
                        self.word_id[self.vocab[token]] = token

                caption = " ".join(tokens)
                self.data.at[i, 'caption'] = caption

                pbar.update(1)

        print "\nVocabulary was successfully built!"

    def pad_data(self):
        """Pads data sequences to the max length"""

        assert self.data is not None, "No data has been loaded yet, call DataHandler.read()"

        print "\nStarting the padding process..."

        with tqdm(total=self.data.shape[0]) as pbar:
            for i, row in self.data.iterrows():
                caption = row['review']
                tokens = word_tokenize(caption)

                for _ in range(self.max_length - len(tokens)):
                    tokens.append(self.PAD)

                caption = " ".join(tokens)
                caption = " ".join([self.START, caption, self.END])

                self.data.at[i, 'caption'] = caption

                pbar.update(1)

        print "\nPadding was successful!"

    def resize_images(self, out_path, size=256):
        """Resizes images in the dataset"""

        assert self.data is not None, "No data has been loaded yet, call DataHandler.read()"

        if not os.path.exists(out_path):
            os.makedirs(out_path)

        print "\nResizing images..."

        with tqdm(total=self.data.shape[0]) as pbar:
            for i, row in self.data.iterrows():
                path = row['filename']
                img_name = path.split('/')[-1]
                image = Image.open(path).convert('RGB')
                image = image.resize([size, size], Image.ANTIALIAS)
                image.save(os.path.join(out_path, img_name), image.format)
                self.data.at[i, 'filename'] = os.path.join(out_path, img_name)

                image = np.array(image, dtype=np.float32)

                print np.mean(image, axis=2).shape

                pbar.update(1)

    # def get_mean(self):
    #     """Returns the mean and standard deviation across all channels for the dataset"""
    #
    #     assert self.data is not None, "No data has been loaded yet, call DataHandler.read()"
    #
    #     if self.mean is not None:
    #         return self.mean
    #
    #     print "\nCalculating mean and standard deviation..."
    #
    #     with tqdm(total=self.data.shape[0]) as pbar:
    #         for i, row in self.data.iterrows():
    #             path = './data/Flickr8k_Dataset/Flicker8k_Dataset/' + row['filename']
    #             image = np.array(Image.open(path).convert('RGB'), dtype=np.float32)
    #             print image.shape

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

        data = {'embeddings': self.embeddings, 'vocab': self.vocab, 'wordmap': self.word_id}

        with gzip.open(out_path, 'wb') as out_file:
            pkl.dump(data, out_file)

        print "\nData stored as {}".format(out_path)

    def load_data(self, path):
        """Loads embeddings and vocab from a zipped pickle file"""

        with gzip.open(path, 'rb') as in_file:
            data = pkl.load(in_file)

        self.embeddings = data['embeddings']
        self.vocab = data['vocab']
        self.word_id = data['wordmap']

        print "\nSuccessfully loaded data from {}".format(path)
