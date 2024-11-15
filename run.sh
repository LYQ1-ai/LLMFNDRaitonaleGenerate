
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

