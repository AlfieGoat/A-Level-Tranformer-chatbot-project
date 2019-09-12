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
"""
token_dict.create_dict([], 100000, 15000, load_tally_dict=True, load_compressed_tally_dict=True)