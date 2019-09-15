import pre_processing_token_dict_creator
import pickle
token_dict = pre_processing_token_dict_creator.TokenDictCreator()
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
with open("current.txt", "w") as file:
    file.write(str(6))
vocab = pickle.load(open("vocab_dict.pickle", "rb"))
import pre_processing_raw_training_data_database
db = pre_processing_raw_training_data_database.ShelfDB()
count = 0
for i in range(1000000):
    comment = db.retrieve_row(i)[0]
    print(count)
