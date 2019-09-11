import pre_processing_token_dict_creator
import pickle

token_dict = pre_processing_token_dict_creator.TokenDictCreator()
token_dict.create_dict(["Z:/Code/A-Level-Tranformer-chatbot-project/data/uncompressed/RC_2019-04"], 100000)
