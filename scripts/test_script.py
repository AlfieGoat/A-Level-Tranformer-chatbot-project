import pre_processing_token_dict_creator
import pickle

token_dict = pre_processing_token_dict_creator.TokenDictCreator()
pickle.dump(token_dict.count_tokens(["Z:/Code/A-Level-Tranformer-chatbot-project/data/uncompressed/RC_2019-05"]),
            open("tally_dict.pickle", "wb"))