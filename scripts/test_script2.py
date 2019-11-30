import pre_processing_token_dict_creator
import pickle

"""
token_dict.count_tokens(["Z:/Code/A-Level-Tranformer-chatbot-project/data/uncompressed/RS_2019-05",
                         "Z:/Code/A-Level-Tranformer-chatbot-project/data/uncompressed/RC_2019-05",
                         "Z:/Code/A-Level-Tranformer-chatbot-project/data/uncompressed/RC_2019-04",
                         "Z:/Code/A-Level-Tranformer-chatbot-project/data/uncompressed/RC_2019-03",
                         "Z:/Code/A-Level-Tranformer-chatbot-project/data/uncompressed/RC_2019-02",
                         "Z:/Code/A-Level-Tranformer-chatbot-project/data/uncompressed/RC_2019-01",
                         "Z:/Code/A-Level-Tranformer-chatbot-project/data/uncompressed/RC_2018-12",
                         "Z:/Code/A-Level-Tranformer-chatbot-project/data/uncompressed/RC_2018-11",
                         "Z:/Code/A-Level-Tranformer-chatbot-project/data/uncompressed/RC_2018-10",
                         "Z:/Code/A-Level-Tranformer-chatbot-project/data/uncompressed/RC_2018-09",
                         "Z:/Code/A-Level-Tranformer-chatbot-project/data/uncompressed/RS_2018-08",
                         "Z:/Code/A-Level-Tranformer-chatbot-project/data/uncompressed/RS_2018-09",
                         "Z:/Code/A-Level-Tranformer-chatbot-project/data/uncompressed/RS_2018-10",
                         "Z:/Code/A-Level-Tranformer-chatbot-project/data/uncompressed/RS_2018-11",
                         "Z:/Code/A-Level-Tranformer-chatbot-project/data/uncompressed/RS_2018-12",
                         "Z:/Code/A-Level-Tranformer-chatbot-project/data/uncompressed/RS_2019-01",
                         "Z:/Code/A-Level-Tranformer-chatbot-project/data/uncompressed/RS_2019-02",
                         "Z:/Code/A-Level-Tranformer-chatbot-project/data/uncompressed/RS_2019-03",
                         "Z:/Code/A-Level-Tranformer-chatbot-project/data/uncompressed/RS_2019-04"])

import pre_processing_convert_to_bpes

print(vocab)
a = " \n\t"
for i in a:
    try:
        print(vocab[i])
    except KeyError:
        vocab[i] = len(vocab)
        pickle.dump(vocab, open("vocab_dict.pickle", "wb"))


import sqlite3




import json
import time
print("lol")
a = pre_processing_token_dict_creator.TokenDictCreator()
vocab_dict = pickle.load(open("tally_dict.pickle", "rb"))
a.compress_vocab(vocab_dict, 15000, 1000)
"""

import pre_processing_raw_train_data_database
import random

db = pre_processing_raw_train_data_database.Database()
token_dict = pickle.load(open("vocab_dict.pickle", "rb"))
token_dict_ids = {}
for key, value in token_dict.items():
    token_dict_ids[value] = key


lb = 0
ub = 1000000000000
old_middle = -1
while True:
    middle = (lb + ub) // 2
    value = db.get_train_data_by_id(middle)
    if middle == old_middle:
        print("lol", middle)
        break
    if value is None:
        ub = middle-1
    else:
        lb = middle+1
    old_middle = middle

while True:
    row = db.get_train_data_by_id(random.randint(0, middle))

    child_tensor = pickle.loads(row[1])
    parent_tensor = pickle.loads(row[2])

    child = []
    parent = []

    for count, value in enumerate(child_tensor):
        child.append(token_dict_ids[value.item()])

    for count, value in enumerate(parent_tensor):
        parent.append(token_dict_ids[value.item()])

    print(f"\nParent({len(parent)}): {''.join(parent)} \nchild({len(child)}): {''.join(child)}")
    print(row[-1], child_tensor.shape)
    print(row[-2], parent_tensor.shape)
    input()



































