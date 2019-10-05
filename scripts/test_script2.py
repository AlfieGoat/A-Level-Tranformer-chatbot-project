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
"""

import sqlite3




import json
import time
print("lol")
a = pre_processing_token_dict_creator.TokenDictCreator()
vocab_dict = pickle.load(open("tally_dict.pickle", "rb"))
a.compress_vocab(vocab_dict, 15000, 1000)






































