import os
import sys
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.nn.utils.rnn import pack_padded_sequence
import torchvision.models as models
from torchvision import transforms
from data_handler import DataHandler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class CaptionNet(nn.Module):
    def __init__(self, config, data_handler=None):
        """Load the pre-trained cnn and replace the last fully connected layer"""
        super(CaptionNet, self).__init__()

        if data_handler is None:
            data_handler = DataHandler()
            data_handler.load_data('./data/processed.pkl.gz')

        if config.encoder_model == 'resnet':
            data_handler.img_size = 224
            encoder = models.resnet101(pretrained='imagenet')
        elif config.encoder_model == 'alexnet':
            data_handler.img_size = 256
            encoder = models.alexnet(pretrained='imagenet')
        elif config.encoder_model == 'vgg':
            data_handler.img_size = 224
            encoder = models.vgg16_bn(pretrained='imagenet')
        elif config.encoder_model == 'squeezenet':
            data_handler.img_size = 224
            encoder = models.squeezenet1_1(pretrained='imagenet')
        else:
            data_handler.img_size = 299
            encoder = models.inception_v3(pretrained='imagenet')

        modules = list(encoder.children())[:-1]
        self.encoder = nn.Sequential(*modules)
        self.linear_en = nn.Linear(encoder.fc.in_features, data_handler.embed_size)
        self.bn = nn.BatchNorm1d(data_handler.embed_size, momentum=0.01)
        self.embed = nn.Embedding(data_handler.embeddings.shape[0], data_handler.embeddings.shape[1])
        self.embed.weight = nn.Parameter(torch.from_numpy(data_handler.embeddings.astype(np.float32)))
        self.lstm = nn.LSTM(data_handler.embed_size, config.hidden_size, config.num_layers, batch_first=True)
        self.linear_de_1 = nn.Linear(config.hidden_size, 512)
        self.linear_de_2 = nn.Linear(512, data_handler.vocab_size)
        self.max_length = data_handler.max_length
        self.data_handler = data_handler
        self.config = config

    def forward(self, images, captions, lengths):
        """Extract feature vectors from input images."""
        with torch.no_grad():
            features = self.encoder(images)
        features = features.reshape(features.size(0), -1)
        features = self.linear_en(features)
        if features.size()[0] != 1:
            features = self.bn(features)
        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True)
        hiddens, _ = self.lstm(packed)
        outputs = self.linear_de_1(hiddens[0])
        outputs = self.linear_de_2(outputs)
        return outputs

    def sample(self, images, states=None):
        """Generate captions for given image using greedy search."""
        sampled_ids = []
        with torch.no_grad():
            features = self.encoder(images)
        features = features.reshape(features.size(0), -1)
        features = self.linear_en(features)
        if features.size()[0] != 1:
            features = self.bn(features)
        inputs = features.unsqueeze(1)
        for i in range(self.max_length):
            hiddens, states = self.lstm(inputs, states)
            outputs = self.linear_de_1(hiddens.squeeze(1))
            outputs = self.linear_de_2(outputs)
            _, predicted = outputs.max(1)
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)
            inputs = inputs.unsqueeze(1)
        sampled_ids = torch.stack(sampled_ids, 1)
        return sampled_ids

    def train(self):
        """Trains the network"""
        if not os.path.exists(self.config.train_dir):
            os.makedirs(self.config.train_dir)

        if self.config.resume:
            print "\nLoading previously trained model..."
            if os.path.isfile(os.path.join(self.config.train_dir, 'save.npy')):
                current_epoch, current_step = np.load(os.path.join(self.config.train_dir, 'save.npy'))
                print current_epoch, "out of", self.config.num_epochs, "epochs completed in previous run."
                try:
                    ckpt_file = os.path.join(self.config.train_dir, "model-{}-{}.ckpt".format(str(current_epoch+1), str(current_step+1)))
                    self.load_state_dict(torch.load(ckpt_file))
                    print "\nResuming training..."
                except Exception as e:
                    print str(e).split('\n')[0]
                    print "\nCheckpoint not found!"
                    sys.exit(0)
            else:
                current_step = 0
                current_epoch = 0
                print "\nCheckpoint not found! Initializing training..."
        else:
            current_step = 0
            current_epoch = 0
            print "\nInitializing training..."

        data_loader = torch.utils.data.DataLoader(dataset=self.data_handler,
                                                  batch_size=self.config.batch_size,
                                                  shuffle=self.config.shuffle,
                                                  num_workers=self.config.num_workers,
                                                  collate_fn=self.data_handler.collate_fn)

        loss_fn = nn.CrossEntropyLoss()
        params = list(self.embed.parameters()) + list(self.lstm.parameters()) + list(self.linear_de_1.parameters()) + list(self.linear_de_2.parameters()) + list(self.linear_en.parameters()) + list(self.bn.parameters())
        optimizer = torch.optim.Adam(params, lr=self.config.learning_rate)

        global_step = current_step
        total_step = len(data_loader)
        for epoch in range(self.config.num_epochs - current_epoch):
            with tqdm(total=self.data_handler.data.shape[0]) as pbar:
                for step, (images, captions, lengths) in enumerate(data_loader):

                    images = images.to(device)
                    captions = captions.to(device)

                    targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]

                    outputs = self.forward(images, captions, lengths)

                    loss = loss_fn(outputs, targets)
                    self.zero_grad()
                    loss.backward()
                    optimizer.step()

                    if step % self.config.log_step == 0:
                        print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'
                              .format(epoch, self.config.num_epochs, step, total_step, loss.item(), np.exp(loss.item())))

                    if (global_step+1) % self.config.save_step == 0:
                        torch.save(self.state_dict(), os.path.join(
                            self.config.train_dir, 'model-{}-{}.ckpt'.format(epoch+1, global_step+1)))
                        np.save(os.path.join(self.config.train_dir, "save"), (epoch, global_step))

                    global_step += 1
                    pbar.update(self.config.batch_size)
                    sys.exit(0)

    def predict(self, image):
        """Runs inference on a sample image and return the caption"""
        if self.config.model_path is 'last':
            if os.path.isfile(os.path.join(self.config.train_dir, 'save.npy')):
                epoch, step = np.load(os.path.join(self.config.train_dir, 'save.npy'))
                try:
                    ckpt_file = os.path.join(self.config.train_dir, "model-{}-{}.ckpt".format(str(epoch+1), str(step+1)))
                    self.load_state_dict(torch.load(ckpt_file))
                    print "\nWeights loaded from {}".format(ckpt_file)
                    found = True
                except Exception as e:
                    print str(e).split('\n')[0]
                    found = False
            else:
                found = False
        else:
            if os.path.isfile(self.config.model_path):
                self.load_state_dict(torch.load(self.config.model_path))
                print "\nWeights loaded from {}".format(self.config.model_path)
                found = True
            else:
                found = False

        if not found:
            print "\nTrained model not found!"
            sys.exit(0)

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.data_handler.mean,
                                 self.data_handler.std)])

        image_resized = image.resize([self.data_handler.img_size, self.data_handler.img_size], Image.LANCZOS)

        if transform is not None:
            image_resized = transform(image_resized).unsqueeze(0)

        image_tensor = image_resized.to(device)

        sampled_ids = self.sample(image_tensor)
        sampled_ids = sampled_ids[0].cpu().numpy()

        sampled_caption = []
        for word_id in sampled_ids:
            word = self.data_handler.word_id[word_id]
            sampled_caption.append(word)
            if word == self.data_handler.END:
                break
        sentence = ' '.join(sampled_caption[1:-1])

        print sentence
        plt.imshow(np.asarray(image))
