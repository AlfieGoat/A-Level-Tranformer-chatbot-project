import pickle


# Takes a string and vocab. It converts the string into its base tokens from the dictionary using the least amount of
# tokens possible, using a nifty recursive function.
def convert_tokens_to_bpes(original_string, vocab):
    tokens_in_string = []
    for key, value in vocab.items():
        if key in original_string:
            tokens_in_string.append(key)

    candidate = []
    for token_in_string in tokens_in_string:
        if original_string[:len(token_in_string)] == token_in_string:
            candidate.append(token_in_string)

    try:
        token_list = [max(candidate, key=len)]
    except ValueError as e:
        print(f"{original_string[0]} not in vocab.")
        vocab[original_string[0]] = len(vocab)
        pickle.dump(vocab, open("vocab_dict.pickle", "wb"))
        return [original_string[0]] + convert_tokens_to_bpes(original_string[1:], vocab)

    if len(token_list[0]) != len(original_string):
        print(token_list)
        token_list += (convert_tokens_to_bpes(original_string[len(token_list[0]):], vocab))
        return token_list
    else:
        return token_list

