import os


from torch.utils.data import DataLoader, Dataset
import pandas as pd
class ImageTextPairDataset(Dataset):

    def __init__(self,dataframe,data_dir):
        self.df = dataframe
        self.image_dir = data_dir + '/images'

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        current_path = os.getcwd()
        image_id,text,label,publish_date = self.df.iloc[idx, 0],self.df.iloc[idx, 1],self.df.iloc[idx, 2],self.df.iloc[idx, 3]
        image_file = f"{current_path}/{self.image_dir}/{image_id}_top_img.png"
        return {
            'id': image_id,
            'image_url':f"file://{image_file}",
            "text":text,
            'label':label,
            "publish_date":publish_date
        }


def load_en_image_text_pair(batch_size = 1):
    data_dir = 'data/ARG_Image_dataset/en'
    file_path = f'{data_dir}/gossipcop.csv'
    df = pd.read_csv(file_path)
    dataset = ImageTextPairDataset(df,data_dir)
    return DataLoader(dataset, batch_size,False,num_workers=4)



if __name__ == '__main__':
    dl = load_en_image_text_pair(2)
    for batch in dl:
        print(batch)
        break