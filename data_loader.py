import os


from torch.utils.data import DataLoader, Dataset
import pandas as pd

import Util

label_dict = {
    'fake':0,
    'real':1,
    'other':2,
    0:'fake',
    1:'real',
    2:'other',
    0.0:'fake',
    1.0:'real',
    2.0:'other'
}


class ImageTextPairDataset(Dataset):

    def __init__(self,dataframe):
        """
        dataframe = {
            'id':,
            'image_url':,
            "text":,
            'label':,
            "publish_date":,
            'image_id':
        }
        """
        self.df = dataframe

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        item = self.df.loc[idx]
        return {
            'id': item.id,
            'image_url': item.image_url,
            "text":item.text,
            'label':item.label,
            "publish_date":item.publish_date,
            'image_id':item.image_id
        }


def load_en_image_text_pair_goss(root_path,batch_size = 1):
    data_dir = root_path
    file_path = f'{data_dir}/gossipcop.csv'
    df = pd.read_csv(file_path)
    df['image_id'] = df['id']
    df['image_url'] = df['id'].map(lambda x : f'file://{root_path}/images/{x}_top_img.png')
    dataset = ImageTextPairDataset(df)
    return DataLoader(dataset, batch_size,False,num_workers=4)

def load_gossipcop_fewshot(few_shot_dir,show_nums=4):
    cs_df = pd.read_csv(f'{few_shot_dir}/cs_shot.csv')
    td_df = pd.read_csv(f'{few_shot_dir}/td_shot.csv')
    return Util.get_few_shot(cs_df,show_nums), Util.get_few_shot(td_df,show_nums)

def get_twitter_image_url_dict(root_path):
    image_dir = f'{root_path}/images'
    return {
       file.split('.')[0] : f'file://{image_dir}/{file}' for file in os.listdir(image_dir)
    }

def load_twitter_data(root_path,batch_size = 1):
    data_dir = root_path
    file_path = f'{data_dir}/twitter.csv'
    df = pd.read_csv(file_path)
    image_id2url_dict = get_twitter_image_url_dict(root_path)

    df = pd.DataFrame({
        'id':df['post_id'],
        'text':df['post_text'],
        'label':df['label'],
        'publish_date':df['timestamp'],
        'image_id':df['image_id'],
        'image_url': df['image_id'].map(lambda x : image_id2url_dict.get(x)),
    })

    dataset = ImageTextPairDataset(df)
    return DataLoader(dataset, batch_size,False,num_workers=4)

def load_politifact(root_path,batch_size=1):
    data_dir = root_path
    file_path = f'{data_dir}/politifact.csv'
    df = pd.read_csv(file_path)
    df = pd.DataFrame({
        'id': df['QueryID'],
        'text': df['QueryText'],
        'label': df['Label'].apply(lambda x : label_dict[x]),
        'publish_date': "",
        'image_id': df['QueryImages'],
        'image_url': df['QueryImages'].apply(lambda x: f'file://{data_dir}/{x}'),
    })
    dataset = ImageTextPairDataset(df)
    return DataLoader(dataset, batch_size,False,num_workers=4)




