import json
from collections import defaultdict
import random
import os

def type2list(dict):
    dict_new = {}
    for id, type in dict.items():
        type = type.strip()
        if '_' in type:
            type = " ".join(type.split('_'))
        if '/' in type:
            type = " ".join(type.split('/'))
        if '-' in type:
            type = " ".join(type.split('-'))
        dict_new[id] = type.lower().split(" ")
    return dict_new

data_path = "../../RE_data/re-tacred"
os.path.join('../../RE_data/re-tacred', 'object_type_tokens.txt')
object_type_tokens = []
with open(os.path.join(data_path, 'train.json'), "r") as f:
    file = json.load(f)
    for line in file:
        object_type_tokens += line["subj_type"].lower().split("_")
        object_type_tokens += line["obj_type"].lower().split("_")

with open(os.path.join(data_path, 'dev.json'), "r") as f:
    file = json.load(f)
    for line in file:
        object_type_tokens += line["subj_type"].lower().split("_")
        object_type_tokens += line["obj_type"].lower().split("_")

with open(os.path.join(data_path, 'test.json'), "r") as f:
    file = json.load(f)
    for line in file:
        object_type_tokens += line["subj_type"].lower().split("_")
        object_type_tokens += line["obj_type"].lower().split("_")

object_type_tokens = list(set(object_type_tokens))

print(len(object_type_tokens))
print(object_type_tokens)
with open(os.path.join(data_path, 'object_type_tokens.txt'), 'w') as f:
    f.write(str(object_type_tokens))
