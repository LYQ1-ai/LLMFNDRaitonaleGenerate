import json
import re

import pandas as pd
import torch
from jupyter_core.version import pattern
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

import os
from tqdm import tqdm

import Util
import data_loader
import pickle

import argparse

from data_loader import load_en_image_text_pair_goss, load_twitter_data, load_politifact

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

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

few_shot_template = """Example: 
news text: <text>{news_text}</text>
output: 
- authenticity: {label}
- reason: {rationale_content}"""

predict_template = """
news text: <text>{news_text}</text>
output: 
"""

prompt_TD = """
The text contained in the <text></text> tag is a news summary.
Please analyze the authenticity of this news article step by step from the perspective of the textual description.
Output in the following format：
- authenticity: a single word: fake or real
- reason: The basis for judging the authenticity of the news from the perspective of the textual description.
Several examples are provided below.
"""


prompt_CS = """
The text contained in the <text></text> tag is a news summary.
Please analyze the authenticity of this news article step by step from the perspective of the common sense.
Output in the following format：
- authenticity: a single word: fake or real
- reason: The basis for judging the authenticity of the news from the perspective of the  .
Several examples are provided below.
"""



prompt_rationales_dict = {
    'td': prompt_TD,
    'cs': prompt_CS,
}

prompt_mode = {
    'td': {'text'},
    'cs': {'text'},
}

dataloader_func_dict = {
    'gossipcop':load_en_image_text_pair_goss,
    'twitter':load_twitter_data,
    'politifact':load_politifact
}

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str)
parser.add_argument('--qwen_path', type=str,default='/home/lyq/Model/Qwen2-VL-7B-Instruct')
parser.add_argument('--root_path', type=str,default='/home/lyq/DataSet/FakeNews/politifact')
parser.add_argument('--few_shot_dir', type=str,default='/home/lyq/DataSet/FakeNews/ARG_Image_dataset/en/few_shot')
args = parser.parse_args()



class MessageUtil:

    def __init__(self,rationale_name):
        self.prompt_template = prompt_rationales_dict[rationale_name]
        self.prompt_mode = prompt_mode[rationale_name]


    def generate_few_shot_msg(self,batch,few_shot):
        """
        batch:{
            'id': item.id,
            'image_url': item.image_url,
            "text":item.text,
            'label':item.label,
            "publish_date":item.publish_date,
            'image_id':item.image_id
        }
        few_shot: {
            'text': ,
            'label': ,
            'rationale': ,
        }
        """
        few_shot = {
            'text':[t[0] for t in few_shot['text']],
            'label':[l[0] for l in few_shot['label']],
            'rationale':[r[0] for r in few_shot['rationale']],
        }

        batch_size = len(batch['id'])
        messages = []
        for i in range(batch_size):
            image_id, url, text, publish_date = batch['id'][i], batch['image_url'][i], batch['text'][i],batch['publish_date'][i]
            msg = {
                "role": "user",
                "content": [{"type": "text", "text": self.prompt_template}]
            }
            nums_few_shot = len(few_shot['text'])
            few_shot_msg = [
                {
                    "type": "text", "text":  few_shot_template.replace('{news_text}',few_shot['text'][i]).replace('{label}',few_shot['label'][i]).replace('{rationale_content}', few_shot['rationale'][i])
                }
                for i in range(nums_few_shot)]
            predict_msg = {"type": "text", "text":  predict_template.format(news_text=text)}
            msg['content'].extend(few_shot_msg)
            msg['content'].append(predict_msg)
            messages.append(msg)

        return messages



def validate_model_output_json(output):
    try:
        json_text = output[0].replace('\n','')
        json_pattern = r'\{.*?\}'
        if not json_text.startswith('{') or not json_text.endswith('}'):
            match = re.search(json_pattern, json_text)
            if not match:
                return {}
            json_text = match.group(0)
        #html_pattern = r'<reason>(.*?)</reason>'
        #json_text = re.sub(html_pattern, replace_quotes, json_text)
        result = json.loads(json_text)

        if 'authenticity' in result and 'reason' in result:
            if result['authenticity'] in label_dict.keys():
                return result
            else:
                result['authenticity']= 'other'
                return result


    except json.JSONDecodeError:
        # 如果解析失败，返回None
        return {}
    return {}

def validate_model_output(output):
    try:
       text = output[0]
       res = {}
       auth,reason = text.split('\n',maxsplit=1)
       if 'fake' in auth.lower():
           res['authenticity'] = 'fake'
       elif 'real' in auth.lower():
           res['authenticity'] = 'real'
       elif 'other' in auth.lower():
           res['authenticity'] = 'other'
       if 'reason:' in reason:
           res['reason'] = reason.split('reason:',maxsplit=1)[1]
       elif 'Reason:' in reason:
           res['reason'] = reason.split('Reason:',maxsplit=1)[1]
       else:
           res['reason'] = None
       return res
    except Exception as e:
        return {}

class Qwen2VL:

    def __init__(self):
        model_dir = args.qwen_path
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_dir, torch_dtype="auto", device_map="auto"
        )

        min_pixels = 256 * 28 * 28
        max_pixels = 1280 * 28 * 28
        self.processor = AutoProcessor.from_pretrained(model_dir, min_pixels=min_pixels, max_pixels=max_pixels)

    def chat(self,messages):
        # Preparation for inference
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True # TODO tokenize=False
        )
        image_inputs, _ = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")
        # Inference: Generation of the output
        generated_ids = self.model.generate(**inputs,max_new_tokens=512,temperature=0.8,top_k=40,top_p=0.9) # max_new_tokens=128,temperature=0.8,top_k=40,top_p=0.9
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text

def generate_LLM_Rationale(data, model, rationale_name,few_shot_iter,cache_file):
    """
    return dict{
        'id':{
            authenticity:str
            reason:str
        }
    }
    """
    msg_util = MessageUtil(rationale_name)
    max_try = 10

    # 检查缓存是否存在并加载
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            ans = pickle.load(f)
    else:
        ans = {}


    for batch in tqdm(data):
        if isinstance(batch['id'], torch.Tensor):
            batch['id'] = batch['id'].tolist()
        batch_id = batch['id'][0]
        if batch_id in ans.keys() and 'reason' in ans[batch_id].keys() and 'reason' in ans[batch_id].keys() and ans[batch_id]['reason'] is not None:
            continue
        few_shot_batch = next(few_shot_iter)
        messages = msg_util.generate_few_shot_msg(batch,few_shot_batch)
        out_dict = {}

        for i in range(max_try):
            out = model.chat(messages)
            print(out[0])
            out_dict = validate_model_output(out)
            if out_dict:  # 有效输出时跳出循环
                break


        ans[batch_id] = out_dict

        # 定期保存缓存
        if len(ans) % 100 == 0:
            with open(cache_file, 'wb') as f:
                pickle.dump(ans, f)

    # 最后一次保存缓存
    with open(cache_file, 'wb') as f:
        pickle.dump(ans, f)

    return ans

def parser_label(rationale_data, index):
    if rationale_data[index] and 'authenticity' in rationale_data[index]:
        label_str = rationale_data[index]['authenticity'].lower()
        if label_str in label_dict.keys():
            return label_dict[label_str]
    return -1


def write_LLM_Rationale(data,data_rationales,save_file_name):
    """
    :param data: {
        'id': item.id,
        'image_url': item.image_url,
        "text":item.text,
        'label':item.label,
        "publish_date":item.publish_date,
        'image_id':item.image_id
    }
    :param data_rationales: dict {
        rationale_name :{
            id :{
                authenticity:str,
                reason:str
            }
        }
    }
    :return:
    """
    result = []
    for batch in data:
        if isinstance(batch['id'], torch.Tensor):
            batch['id'] = batch['id'].tolist()
        image_id, url, text, publish_date,label,text_id = batch['image_id'][0], batch['image_url'][0], batch['text'][0],batch['publish_date'][0],batch['label'][0],batch['id'][0]
        content, label, time, source_id, split = text,label,publish_date,text_id,None
        item_dict = {
            'content':content,
            'label':label_dict[label],
            #'publish_date':publish_date,
            'image_id':image_id,
            'source_id':source_id,
            'split':split,
        }

        for rationale_name in prompt_rationales_dict.keys():
            rationale_item = data_rationales[rationale_name][source_id]
            if not rationale_item:
                continue
            item_dict[f'{rationale_name}_rationale'] = data_rationales[rationale_name][source_id]['reason']
            item_dict[f'{rationale_name}_pred'] = label_dict[data_rationales[rationale_name][source_id]['authenticity']]
            item_dict[f'{rationale_name}_acc'] = int(item_dict[f'{rationale_name}_pred'] == item_dict['label'])

        result.append(item_dict)

    pd.DataFrame(result).to_csv(save_file_name, index=False)





if __name__ == '__main__':



    Qwen2VL = Qwen2VL()

    data_name = args.dataset
    print(f'generate : {data_name}')

    data = dataloader_func_dict[data_name](root_path=args.root_path)
    cs_shot_df ,td_shot_df = data_loader.load_gossipcop_fewshot(args.few_shot_dir,4)

    few_shot_df_dict = {
        'td':td_shot_df,
        'cs':cs_shot_df
    }

    data_rationales = {}
    for rationale_name in prompt_rationales_dict.keys():
        cache_file = f'cache/{data_name}/{rationale_name}.pkl'
        print(f"start generate {rationale_name} data.............")
        generate_LLM_Rationale(data,Qwen2VL,rationale_name,few_shot_df_dict[rationale_name],cache_file)

    for rationale_name in prompt_rationales_dict.keys():
        cache_file = f'cache/{data_name}/{rationale_name}.pkl'
        with open(cache_file, 'rb') as f:
            data_rationales[rationale_name] = pickle.load(f)

    save_file = f'/home/lyq/DataSet/FakeNews/twitter_dataset/{data_name}_llm_rationales.csv'
    write_LLM_Rationale(data,data_rationales,save_file)
    Util.calculate_acc(pd.read_csv(save_file))




















