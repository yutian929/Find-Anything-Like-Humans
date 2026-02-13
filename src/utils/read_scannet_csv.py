import csv
import json
from collections import OrderedDict


def read_csv_and_convert_to_dict(csv_path: str, json_path: str):
    # 创建一个空字典来存储数据
    category_dict = {}

    # 打开CSV文件并读取
    with open(csv_path, mode="r", encoding="utf-8") as file:
        # 指定分隔符为制表符（tab）
        reader = csv.reader(file, delimiter="\t")
        next(reader)  # 跳过标题行

        for row in reader:
            # 读取ID、raw_category和category列
            id_value = int(row[0])  # ID 作为字典的键
            raw_category = row[1]
            category = row[2]

            # 将数据存储到字典中
            category_dict[id_value] = {
                "raw_category": raw_category,
                "category": category,
            }

    # 按ID进行排序，并返回一个按升序排列的OrderedDict
    sorted_dict = OrderedDict(sorted(category_dict.items()))

    # 将字典保存为JSON文件
    with open(json_path, "w", encoding="utf-8") as json_file:
        json.dump(sorted_dict, json_file, ensure_ascii=False, indent=4)

    print(f"Data has been saved to {json_path}")


# 示例用法
csv_path = "/home/yutian/下载/scannetv2/scannetv2-labels.combined.tsv"  # 替换为你的 CSV 文件路径
json_path = "scannetv2_id_category.json"  # 替换为你想保存的 JSON 文件路径

read_csv_and_convert_to_dict(csv_path, json_path)
