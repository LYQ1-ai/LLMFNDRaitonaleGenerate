import itertools
from abc import abstractmethod

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm import tqdm

from data_loader import load_gossipcop_fewshot

label_dict = {
    'real':1,
    'fake':0,
    1:'real',
    0:'fake'
}


prompt_mode = {
    'td': {'text'},
    'cs': {'text','image'},
}

class LLMPredictDataset(Dataset):

    def __init__(self,df):
        self.df = df

    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.loc[idx]
        result = {
                'text': row['text'].tolist(),
                'label': row['label'].apply(lambda x: label_dict[x]).tolist(),
            }
        if 'rationale' in self.df.columns:
            result['rationale'] = row['rationale'].tolist()
        return result


class BalancedSampler(Sampler):
    def __init__(self, labels, batch_size=10):
        self.labels = np.array(labels)
        self.batch_size = batch_size
        self.fake_indices = np.where(self.labels == 0)[0]
        self.real_indices = np.where(self.labels == 1)[0]

    def __iter__(self):
        num_batches = len(self.labels) // self.batch_size
        for _ in range(num_batches):
            # 随机抽取5个fake和5个real
            real_nums = self.batch_size // 2
            fake_nums = self.batch_size - real_nums
            fake_batch = np.random.choice(self.fake_indices, real_nums, replace=False)
            real_batch = np.random.choice(self.real_indices, fake_nums, replace=False)
            batch = np.concatenate((fake_batch, real_batch))
            np.random.shuffle(batch)
            yield batch  # 返回该批次的索引

    def __len__(self):
        return len(self.labels) // self.batch_size


class LLM:

    few_shot_prompt = """
The text encompassed by the tags <text></text> is a title of the news.
Please make a judgment on the authenticity of the news.
the output should contain only one word: real or fake.
Several examples are provided below.
"""
    few_example = """<text>{news_text}</text>  
            {label}"""



    @abstractmethod
    def chat(self,prompt,history):
        raise NotImplementedError


    def text2Msg(self,text):
        return {
            'role':'user',
            'content':text
        }

    def predict(self,news,few_shot_examples:dict[str,list[str]])->int:
        news_msg = self.few_example.format(news_text=news,label=' ')
        history = [self.text2Msg(self.few_shot_prompt)]
        few_shot_nums = len(few_shot_examples['label'])
        for i in range(few_shot_nums):
            text = few_shot_examples['text'][i]
            label = few_shot_examples['label'][i]
            history.append(self.text2Msg(self.few_example.format(news_text=text,label=label)))


        max_try = 10
        for i in range(max_try):
            response = self.chat(news_msg, history)
            print(response)
            if response in label_dict.keys():
                return label_dict[response]
        return -1


def get_few_shot(df,shot_nums=5):
    few_shot_df = df
    llm_ds = LLMPredictDataset(few_shot_df)
    sampler = BalancedSampler(df['label'],shot_nums)
    return itertools.cycle(iter(DataLoader(llm_ds,sampler=sampler)))




def generate_llm_predict(model,pred_df,few_shot_df, shot_nums=5):
    few_shot_iter = get_few_shot(few_shot_df,shot_nums)
    result = pred_df
    for i in tqdm(range(len(result))):
        line = result.loc[i,'line']
        print(f"predict line :{line}")
        news = result.loc[i,'text']
        pred = model.predict(news,next(few_shot_iter))
        result.loc[i,'llm_pred'] = pred
        result.loc[i,'llm_acc'] = int(pred == result.loc[i,'label'])

    result = result[result['llm_pred'] != -1]
    return result


def calculate_acc(df):
    print(f"sum data {df.shape[0]}")
    for rationale_name in prompt_mode.keys():
        legal_data_df = df[(df[f'{rationale_name}_pred']!=-1) & (df[f'{rationale_name}_rationale'] is not None)]
        print(f"{rationale_name} : acc {(legal_data_df[f'{rationale_name}_acc'] == 1).sum() / legal_data_df.shape[0]} , "
              f" acc_real {((legal_data_df[f'{rationale_name}_acc'] == 1) & (legal_data_df['label'] == 1)).sum() / (legal_data_df['label'] == 1).sum()} ,"
              f" acc_fake {((legal_data_df[f'{rationale_name}_acc'] == 1) & (legal_data_df['label'] == 0)).sum() / (legal_data_df['label'] == 0).sum()} ,")


