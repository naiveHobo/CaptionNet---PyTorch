import pandas as pd
import os

def load_flickr(img_dir, annot_path):
    with open(annot_path, 'r') as flickr:
        data = flickr.readlines()

    df = pd.DataFrame()

    filenames = []
    captions = []

    for line in data:
        splits = line.replace('\n', '').split('\t')
        if int(splits[0].split('#')[1]) == 4:
            filenames.append(os.path.join(img_dir, splits[0].split('#')[0]))
            captions.append(splits[1])

    df['filename'] = filenames
    df['caption'] = captions

    return df
