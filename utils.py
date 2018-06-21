import pandas as pd


def load_flickr(path):
    with open(path, 'r') as flickr:
        data = flickr.readlines()

    df = pd.DataFrame()

    filenames = []
    captions = []

    for line in data:
        splits = line.replace('\n', '').split('\t')
        if int(splits[0].split('#')[1]) == 4:
            filenames.append('./data/Flickr8k_Dataset/Flicker8k_Dataset/'+splits[0].split('#')[0])
            captions.append(splits[1])

    df['filename'] = filenames
    df['caption'] = captions

    return df
