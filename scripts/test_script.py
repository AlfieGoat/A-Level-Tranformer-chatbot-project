"""
import pre_processing_token_dict_creator
import pickle


token_dict = pre_processing_token_dict_creator.TokenDictCreator()
# token_dict.create_dict(["Z:/Code/A-Level-Tranformer-chatbot-project/data/uncompressed/RC_2019-04"], 100000)
vocab_dict = pickle.load(open("vocab_dict.pickle", "rb"))

print(token_dict.count_bpe_tokens(vocab_dict, return_list=True))
"""

a = "hello£$%^&😋"
old_string = str(a.encode("utf-8"))
new_string = ""
for count, value in enumerate(old_string):
    if count != 0 and count != 1 and count != len(old_string)-1:
        new_string += value
# b = b.encode("ascii")
print(new_string)
print(eval("b'"+new_string+"\xf0\x9f\x98\x8b'").decode('utf-8'))
