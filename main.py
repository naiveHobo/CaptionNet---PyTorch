import os
import argparse
from model import CaptionNet
from data_handler import DataHandler
from utils import load_flickr
import torch
from torchvision import transforms
from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ap = argparse.ArgumentParser()

ap.add_argument("--mode", type=str, help="train|test",
                choices=["train", "test"], required=True)
ap.add_argument("--resume", action="store_true",
                help="resume training from last checkpoint")
ap.add_argument("--train_dir", default="./train",
                help="path to the directory where checkpoints are to be stored")
ap.add_argument("--img_dir", default="./data/Flickr8k_Dataset/",
                help="path to the directory where images are stored")
ap.add_argument("--annot_path", default="./data/Flickr8k_text/Flickr8k.token.txt",
                help="path to annotation file")
ap.add_argument("--model_path", default="./model/model.ckpt",
                help="path to ckpt file (loads the last trained model is not set)")
ap.add_argument("--sample", default="./sample/test.jpg",
                help="path to the directory where checkpoints are to be stored")
ap.add_argument("--num_epochs", type=int, default=10,
                help="number of epochs")
ap.add_argument("--batch_size", type=int, default=8,
                help="size of mini-batch")
ap.add_argument("--learning_rate", type=float, default=0.001,
                help="learning rate")
ap.add_argument('--log_step', type=int, default=10,
                help='step size for printing log info')
ap.add_argument('--save_step', type=int, default=1000,
                help='step size for saving trained models')
ap.add_argument('--hidden_size', type=int, default=128,
                help='dimension of lstm hidden states')
ap.add_argument('--num_layers', type=int, default=1,
                help='number of layers in lstm')
ap.add_argument('--shuffle', action="store_true", help="shuffle datatset")
ap.add_argument('--num_workers', type=int, default=2)

args = ap.parse_args()

input_size = {
    'alexnet': 224,
    'densenet': 224,
    'resnet': 224,
    'inception': 299,
    'squeezenet': 224,
    'vgg': 224
}

if args.mode == "train":
    flickr = load_flickr(args.img_dir, args.annot_path)
    data = DataHandler(flickr)

    if os.path.isfile('./data/processed.pkl.gz'):
        data.load_data('./data/processed.pkl.gz')
    else:
        data.build_vocab()

    data.resize_images('./data/resized', img_size=input_size['resnet'])
    data.pad_data()

    if not os.path.isfile('./data/processed.pkl.gz'):
        data.save_data('./data/processed.pkl.gz')

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])

    data.set_augment(transform)

    net = CaptionNet(data_handler=data, config=args).to(device)
    net.train_model()

if args.mode == "test":
    image = Image.open(args.sample).convert('RGB')
    net = CaptionNet(config=args).to(device)
    net.predict(image)
    # net.export_to_onnx()
