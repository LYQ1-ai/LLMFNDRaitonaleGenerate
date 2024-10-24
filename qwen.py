import json
import re

import pandas as pd
import torch
from jupyter_core.version import pattern
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

import os
from tqdm import tqdm
import data_loader
import pickle

label_dict = {
    'fake':0,
    'real':1,
}

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'




prompt_TD = """
The text enclosed in the <text></text> tags is a news summary.
Please analyze the authenticity of this news article step by step from the perspective of the textual description.
Output the results in JSON format as a single line, with the following example structure: {"authenticity": "a single word: fake or real","reason": "The basis for judging the authenticity of the news from the perspective of the textual description."}
news text: <text>{news text}</text>
"""

prompt_IA = """
The given image is the cover of a news article.
Please analyze the authenticity of this news article step by step from the perspective of whether the image has been edited or manipulated.
Output the results in JSON format as a single line, with the following example structure: {"authenticity": "a single word: fake or real","reason": "The basis for judging the authenticity of the news from the perspective of whether the image has been edited or manipulated."}
"""

prompt_ITC = """
The text enclosed in the <text></text> tags is a news summary, and the given image is the cover of that news article.
Please analyze the authenticity of this news article step by step from the perspective of whether there are contradictions between the image and the textual description.
Output the results in JSON format as a single line, with the following example structure: {"authenticity": "a single word: fake or real","reason": "the basis for determining the authenticity of the news from the perspective of whether there are contradictions between the image and the textual description."}
news text: <text>{news text}</text>
"""

prompt_CS = """
The text enclosed in the <text></text> tags is a news summary, and the given image is the cover of that news article.
Please analyze the authenticity of this news article step by step from the perspective of common sense, considering both the text and the given image.
Output the results in JSON format as a single line, with the following example structure: {"authenticity": "a single word: fake or real","reason": "the basis for determining the authenticity of the news from the perspective of common sense, considering both the text and the given image."}
news text: <text>{news text}</text>
"""


prompt_rationales_dict = {
    'td': prompt_TD,
    'ia': prompt_IA,
    'cs': prompt_CS,
    'itc': prompt_ITC
}

prompt_mode = {
    'td': {'text'},
    'ia': {'image'},
    'cs': {'text','image'},
    'itc': {'text','image'}
}



class MessageUtil:

    def __init__(self,rationale_name):
        self.prompt_template = prompt_rationales_dict[rationale_name]
        self.prompt_mode = prompt_mode[rationale_name]



    def generate_msg(self,batch):
        """
        :param batch: [url_tuple,text_tuple] , url_tuple = (url1,url2), text_tuple = (text1,text2)
        :return: messages:list(dict)
        """
        batch_size = len(batch['id'])
        messages = []
        for i in range(batch_size):
            image_id, url, text,publish_date = batch['id'][i], batch['image_url'][i], batch['text'][i],batch['publish_date'][i]
            msg = {
                    "role": "user",
                    "content": [{"type": "text", "text": self.prompt_template}]
            }
            if 'text' in self.prompt_mode:
                msg['content'][0] = {"type": "text", "text": self.prompt_template.replace('{news text}', text)}

            if 'image' in self.prompt_mode:
                msg['content'].append({
                    "type": "image",
                    "image": url,
                })
            messages.append(msg)

        return messages


def validate_model_output(output):
    try:
        json_text = output[0].replace('\n','')
        json_pattern = r'\{.*?\}'
        if not json_text.startswith('{') or not json_text.endswith('}'):
            match = re.search(json_pattern, json_text)
            if not match:
                return {}
            json_text = match.group(0)
        # 尝试将JSON字符串解析为字典
        # json_text = json_text.replace("'", '"')
        result = json.loads(json_text)

        if 'authenticity' in result and 'reason' in result:
            return result
    except json.JSONDecodeError:
        # 如果解析失败，返回None
        return {}
    return {}


class Qwen2VL:

    def __init__(self):
        model_dir = 'Qwen2-VL-7B-Instruct'
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
        generated_ids = self.model.generate(**inputs,max_new_tokens=512) # max_new_tokens=128,temperature=0.8,top_k=40,top_p=0.9
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text

def generate_LLM_Rationale(data, model, rationale_name):
    msg_util = MessageUtil(rationale_name)
    max_try = 10
    cache_file_path = f'cache/{rationale_name}.pkl'

    # 检查缓存是否存在并加载
    if os.path.exists(cache_file_path):
        with open(cache_file_path, 'rb') as cache_file:
            ans = pickle.load(cache_file)
    else:
        ans = []

    data = [batch for batch in data]
    if len(data) == len(ans):
        return ans

    # 处理未生成的部分
    data = data[len(ans):]

    for batch in tqdm(data):
        messages = msg_util.generate_msg(batch)
        out_dict = {}

        for i in range(max_try):
            out = model.chat(messages)
            print(out[0])
            out_dict = validate_model_output(out)
            if out_dict:  # 有效输出时跳出循环
                break

        ans.append(out_dict)

        # 定期保存缓存
        if len(ans) % 100 == 0:
            with open(cache_file_path, 'wb') as cache_file:
                pickle.dump(ans, cache_file)

    # 最后一次保存缓存
    with open(cache_file_path, 'wb') as cache_file:
        pickle.dump(ans, cache_file)

    return ans

def parser_label(rationale_data, index):
    if rationale_data[index] and 'authenticity' in rationale_data[index]:
        label_str = rationale_data[index]['authenticity'].lower()
        if label_str in label_dict.keys():
            return label_dict[label_str]
    return -1


def write_LLM_Rationale(data,data_rationales):
    """
    :param data:
    :param data_rationales: dict {'rationales_name':list(dict),}
    :return:
    """
    data_list = []
    for batch in data:
        batch_size = len(batch['id'])
        for i in range(batch_size):
            data_list.append({
                'content':batch['text'][i],
                'label':label_dict[batch['label'][i]],
                'time':batch['publish_date'][i],
                'source_id':batch['id'][i],
                'split':None
            })
    for rationale_name in data_rationales.keys():
        data_rationale = data_rationales[rationale_name]
        assert len(data_list) == len(data_rationale)
        for i in range(len(data_list)):
            rationale_label = parser_label(data_rationale, i)
            data_list[i][rationale_name] = data_rationale[i]['reason'] if data_rationale[i] else None
            data_list[i][f'{rationale_name}_pred'] = rationale_label
            data_list[i][f'{rationale_name}_acc'] = int(rationale_label == data_list[i]['label']) if data_rationale[i] else -1
    df = pd.DataFrame(data_list)
    df.to_csv('data/ARG_Image_dataset/en/gossipcop_llm_rationales.csv',index=False)






if __name__ == '__main__':
    Qwen2VL = Qwen2VL()
    data = data_loader.load_en_image_text_pair()
    data_rationales = {}
    for rationale_name in prompt_rationales_dict.keys():
        print(f"start generate {rationale_name} data.............")
        generate_LLM_Rationale(data,Qwen2VL,rationale_name)

    for rationale_name in prompt_rationales_dict.keys():
        with open(f'cache/{rationale_name}.pkl', 'rb') as f:
            data_rationales[rationale_name] = pickle.load(f)

    write_LLM_Rationale(data,data_rationales)


















