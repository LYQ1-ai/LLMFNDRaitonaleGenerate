# LLM FND Rationale Generate

- 本仓库用于通过LLM判断新闻真假性并生成Rationale,基本架构如下

  

![](/home/lyq/PycharmProjects/llamaRationaleGenerate/LLMFNDRationaleGenerate.png)



- 环境搭建

  - 基本环境：CUDA 12.1，Python=3.10.15

  ```shell
  conda env create -f env.yaml
  ```

- ## 项目结构总览

  - ```shell
    .
    ├── cache # 用来持久化LLM生成的数据，每生成100条会保存一次，如果程序崩溃可以快速恢复之前生成的结果，如果需要重新生成需要删除相应数据集的缓存
    │   ├── gossipcop
    │   │   ├── cs.pkl # cs开头为从常识角度进行分析，td为从文字描述角度进行分析
    │   │   └── td.pkl
    │   ├── politifact
    │   │   ├── cs.pkl
    │   │   └── td.pkl
    │   └── twitter
    │       ├── cs.pkl
    │       └── td.pkl
    ├── data # 数据集文件夹，输入和最终输出均在此文件夹
    │   ├── gossipcop
    │   │   ├── data_processor.ipynb # 数据预处理脚本
    │   │   ├── gossipcop.csv# 数据预处理得到的结果
    │   │   ├── gossipcop_llm_rationales.csv # 最终输出分割后的结果
    │   │   ├── rationale_data_processor.ipynb# 数据后处理脚本
    │   │   ├── test.csv #最终输出数据集分割后的结果
    │   │   ├── train.csv
    │   │   └── val.csv
    │   ├── politifact
    │   │   ├── data_processor.ipynb
    │   │   ├── politifact.csv
    │   │   └── politifact_llm_rationales.csv
    │   └── twitter
    │       ├── data_processor.ipynb
    │       ├── rationale_data_processor.ipynb
    │       ├── test.csv
    │       ├── train.csv
    │       ├── twitter.csv
    │       ├── twitter_llm_rationales.csv
    │       └── val.csv
    ├── data_loader.py #加载原始数据集脚本
    ├── env.yaml #环境依赖声明
    ├──  .gitignore
    ├── LLMFNDRationaleGenerate.png
    ├── prompt 
    ├── qwen.py #LLM生成脚本
    ├── README.md
    ├── run.sh #项目运行脚本
    └── Util.py # 一些工具函数
    ```

- ## 数据集

  - gossipcop

    - 原始数据集可以通过[链接](https://drive.google.com/drive/folders/1rLrh5x5UlYskfbhhVyz523MKgmCDyuX2)获取(下载gossipcop_v3_origin.json)，相关图片可以通过[链接](https://drive.google.com/drive/folders/11okt9IRDxXgfTr7Ae1wxl9CHZC1PphhC)获取

    - 下载原始数据集后整理成以下结构

      ```shell
      .
      ├── gossipcop_v3_origin.json # 原始数据集
      ├── images # 图片文件夹
      ```

  - twitter

    - 原始数据集通过该[仓库](https://github.com/plw-study/MRML)可以找到下载地址

    - 下载原始数据集后整理成以下结构

      ```shell
      .
      ├── devset
      ├── testset
      ```

    

- ## 预处理数据集

  下面以gossipcop为例进行数据的预处理

  - gossipcop

    - 通过data_processor.ipynb对gossipcop_v3_origin.json进行预处理得到gossipcop.csv

    - **注意：请自行提供few_shot数据(csv格式)并指定few_shot所在的文件夹，基本结构如下**

      ```shell
      .
      ├── cs_shot.csv 
      └── td_shot.csv
      
      (这里使用json格式方便解释，json中的key对应csv中的列，实际上还是csv结构)
      cs_shot.csv 结构如下：
      {
      	"text":"news text",
      	"rationale": "rationale text",
      	"label": "groud truth"
      }
      ```

- ## 运行

  - 运行qwen.py脚本（或者自行修改run.sh并执行run.sh）

    ```shell
    
    # --dataset:选用的数据集
    # --qwen_path:本地Qwen模型路径
    # --root_path:数据集文件目录路径
    # --few_shot_dir:few_shot数据目录路径
    # --few_shot_nums:提供的few-shot数量
    python qwen.py --dataset politifact \
                   --qwen_path /the/path/of/qwen \
                   --root_path /the/path/of/you/dataset \
                   --few_shot_dir /the/path/of/you/few_shot_data \
                   --few_shot_nums 4
    ```

  - 运行结束后得到gossipcop_llm_rationales.csv，gossipcop.csv结构如下

    ```shell
    {
        "content": "news text",
        "label": "0 or 1,news label",
        "source_id": "text_id",
        "image_id": "image_id,use to find image file",
        "td_rationale": "rationale from the perspective of the textual description",
        "td_pred": "1 or 0,llm pred news real or fake from the perspective of the textual description",
        "td_acc": "1 or 0,llm pred Right or wrong from the perspective of the textual description",
        "cs_rationale": "rationale from the perspective of the common sense",
        "cs_pred": "1 or 0,llm pred news real or fake from the perspective of the common sense",
        "cs_acc": "1 or 0,llm pred Right or wrong from the perspective of the common sense",
        "split": "train or val or test"
      },
    ```

  - 通过rationale_data_processor.ipynb对gossipcop_llm_rationales.csv进行分割和过滤得到，最终文件结构如下

    ```shell
    .
    ├── data_processor.ipynb
    ├── gossipcop.csv
    ├── gossipcop_llm_rationales.csv
    ├── gossipcop_v3_origin.json
    ├── images
    ├── rationale_data_processor.ipynb
    ├── test.csv
    ├── train.csv
    └── val.csv
    ```

  

  

  

  

  

  

  

















四种数据集数据分布

| 数据集          | 总量 (sum) | 真数据量 (real data) | 假数据量 (fake data) | 真数据占比 (%) | 假数据占比 (%) |
| --------------- | ---------- | -------------------- | -------------------- | -------------- | -------------- |
| gpt_gossipcop   | 6416       | 1484                 | 4932                 | 23.13%         | 76.87%         |
| qwen_gossipcop  | 12332      | 9616                 | 2716                 | 77.96%         | 22.04%         |
| qwen_twitter    | 14195      | 6432                 | 7763                 | 45.30%         | 54.70%         |
| qwen_politifact | 2025       | 0                    | 2025                 | 0.00%          | 100.00%        |

gpt_gossipcop

|      | acc    | acc_real | acc_fake |
| ---- | ------ | -------- | -------- |
| td   | 0.7593 | 0.3564   | 0.8805   |
| cs   | 0.8145 | 0.4993   | 0.9093   |

qwen_gossipcop

|      | acc    | acc_real | acc_fake |
| ---- | ------ | -------- | -------- |
| td   | 0.7814 | 0.9499   | 0.1848   |
| cs   | 0.7851 | 0.9595   | 0.1678   |

qwen_twitter

|      | acc    | acc_real | acc_fake |
| ---- | ------ | -------- | -------- |
| td   | 0.5785 | 0.3787   | 0.7440   |
| cs   | 0.5692 | 0.4528   | 0.6655   |

qwen_politifact

|      | acc    | acc_real | acc_fake |
| ---- | ------ | -------- | -------- |
| td   | 0.7901 | -        | 0.7901   |
| cs   | 0.7367 | -        | 0.7367   |



