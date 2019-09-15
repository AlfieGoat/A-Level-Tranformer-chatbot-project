import sqlite3
from pre_processing_database import Database
import pre_processing_token_dict_creator
import pickle
import time
import ctypes

def convert_tokens_to_bpes(original_string):
    token_list = []
    working_string = original_string
    tokens_in_string = []
    for key in vocab_list:
        if key in working_string:  # Goes over the vocab and sees which tokens from the vocab are in the og string
            tokens_in_string.append(key)

    tokens_in_string = sorted(tokens_in_string, key=len)[::-1] # negligible

    while len(working_string) != 0:
        start = time.time()
        candidate = ""
        for token_in_string in tokens_in_string:
            if working_string[:len(token_in_string)] == token_in_string:  # Checks if token can be used
                candidate = token_in_string
                break

        times.append((time.time() - start))
        if candidate != "":
            token_list.append(candidate)
            working_string = working_string[len(candidate):]
        else:  # Validation: Value error is raised when something is not in the vocab
            print(f"{working_string[0]} not in vocab.")
            return None



    return token_list
times = []
vocab = pickle.load(open("vocab_dict.pickle", "rb"))
"""
vocab_list = []
for key, value in vocab.items():
    vocab_list.append(key)
#vocab_list = sorted(vocab_list, key=len)[::-1]

start = time.time()
for i in range(200):
    (convert_tokens_to_bpes("I'm currently training a certain self-consistent GAN that models noise in an unsupervised way. I have scanned document images that are passed through a generator to output the noise map (scan noise) and this is subtracted from the original scanned document image to output a clean version of the scanned image. This clean version is compared with actual clean document images in an adversarial way. Since my images are roughly 468x468, what is the best way to architect a discriminator? I have just about 8k images each (scanned and clean). The generator uses 3x3 kernels to maintain the text resolution, but the discriminator must downsample inherently. Wouldn't using large kernel sizes or strides destroy the minute scan noise?"))
print(time.time()-start)
count = 0
for i in times:
    count+=i
print(count)
"""
og_vocab = ["o", "lol", "hmm","he","h","e","aw","l"]
"""

vocab_type = ctypes.c_wchar_p * 8
vocab = vocab_type(*og_vocab)
pointer = (c_lib.convert_tokens_to_bpes(5, "hello", vocab, 8))
print(ctypes.POINTER(ctypes.c_wchar_p(pointer)))
"""
# The vocab
vocab_list = []
length = 0
val = ""
for key, value in vocab.items():
    vocab_list.append(key)
    if len(key) > length:
        val = key
        length = len(key)
#vocab_list = sorted(vocab_list, key=len)[::-1]
og_vocab = ["o", "lol", "hmm", "hel", "h", "e", "aw", "l"]
vocab = (ctypes.c_char_p * len(vocab_list))()
vocab[:] = [bytes(x, "utf-8") for x in vocab_list]


word = """as a string instead o=f byte objcthrtg which"""
print(len(bytes(word, "utf-8")))
word_to_go = (ctypes.c_char_p)()
word_to_go.value = bytes(word, "utf-8")
print(bytes(word, "utf-8"))

start = time.time()
# ctypes stuff
for i in range(100):
    c_lib = ctypes.CDLL("Z:/Code/A-Level-Tranformer-chatbot-project/scripts/bpe.dll")
    convert_tokens_to_bpes = c_lib.convert_tokens_to_bpes
    convert_tokens_to_bpes.argtypes = (ctypes.c_int, ctypes.c_char_p, ctypes.c_char_p * len(vocab_list), ctypes.c_int)
    convert_tokens_to_bpes.restype = ctypes.POINTER(ctypes.c_char_p)
    result = convert_tokens_to_bpes(len(bytes(word, "utf-8")), word_to_go, vocab, len(vocab_list))
for i in result:
    print(i)



