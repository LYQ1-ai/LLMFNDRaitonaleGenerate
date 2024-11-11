import os


from torch.utils.data import DataLoader, Dataset
import pandas as pd

import Util


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


def load_en_image_text_pair_goss(batch_size = 1):
    data_dir = '/home/lyq/DataSet/FakeNews/ARG_Image_dataset/en'
    file_path = f'{data_dir}/gossipcop.csv'
    df = pd.read_csv(file_path)
    df['image_id'] = df['id']
    df['image_url'] = df['id'].map(lambda x : f'file:///home/lyq/DataSet/FakeNews/ARG_Image_dataset/en/images/{x}_top_img.png')
    dataset = ImageTextPairDataset(df)
    return DataLoader(dataset, batch_size,False,num_workers=4)

def load_gossipcop_fewshot(show_nums=4):
    cs_df = pd.read_csv('/home/lyq/DataSet/FakeNews/ARG_Image_dataset/en/few_shot/cs_shot.csv')
    td_df = pd.read_csv('/home/lyq/DataSet/FakeNews/ARG_Image_dataset/en/few_shot/td_shot.csv')
    return Util.get_few_shot(cs_df,show_nums), Util.get_few_shot(td_df,show_nums)

def get_twitter_image_url_dict():
    image_dir = '/home/lyq/DataSet/FakeNews/twitter_dataset/images'
    return {
       file.split('.')[0] : f'file://{image_dir}/file' for file in os.listdir(image_dir)
    }

def load_twitter_data(batch_size = 1):
    data_dir = '/home/lyq/DataSet/FakeNews/twitter_dataset'
    file_path = f'{data_dir}/twitter.csv'
    df = pd.read_csv(file_path)
    image_id2url_dict = get_twitter_image_url_dict()

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

if __name__ == '__main__':
    dl = load_en_image_text_pair(2)
    for batch in dl:
        print(batch)
        break