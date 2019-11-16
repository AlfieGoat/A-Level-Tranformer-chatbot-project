import pickle

# Takes a string and vocab. It converts the string into its base tokens
# from the dictionary using the least amount of
# tokens possible, using a nifty recursive function.


def convert_tokens_to_bpes(original_string, vocab_list):
    token_list = []
    working_string = original_string

    tokens_in_string = []
    for key in vocab_list:
        if key in working_string:  # Goes over the vocab and sees which tokens from the vocab are in the og string
            tokens_in_string.append(key)

    tokens_in_string = sorted(tokens_in_string, key=len)[::-1]
    while len(working_string) != 0:
        candidate = ""
        for token_in_string in tokens_in_string:
            if working_string[:len(token_in_string)] == token_in_string:  # Checks if token can be used
                candidate = token_in_string
                break
        if candidate != "":
            token_list.append(candidate)
            working_string = working_string[len(candidate):]
        else:  # Validation: Value error is raised when something is not in the vocab
            # print(f"{working_string[0]} not in vocab.")
            return None

    return token_list


"""
example of how ^ this works:
original_string = endeavour      vocab = [e,n,d,a,v,o,u,r,end,eav,our]
it will take e and see if original_string[:len(e)] == e
which in this case is the same as: e == e which is true:
so we add e to the candidates.
then same for n: original_string[:len(n)] == n
which is the same as: e == n: which is false.
It goes through all of them, but it is interesting when we get to the longer tokens:
original_string[:len(end)] == end
which is the same as end == end which is true
so it adds end to the candidates.
Once we have iterated through all of the valid tokens, we pick the longest one.
In this case it would be end.
We check if end is the same length as the original token, but its not.
So we remove the end from the original token, then send the rest of the token 
the same function recursively which will find the next token and then same again until the 
whole token has been found.
endeavour will turn into [end, eav, our]    
"""


