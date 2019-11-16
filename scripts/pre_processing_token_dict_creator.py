import json
import pickle
import time


class TokenDictCreator:

    def create_dict(self, file_paths,
                    pre_bpe_vocab_size, vocab_size,
                    filter_parameters=(2, ["deleted", "removed"]),
                    track_progress=2000000, load_tally_dict=False,
                    load_compressed_tally_dict=False):
        # track_progress is the num of rows to process before
        # giving elapsed time. Set to 0 for no tracking
        # filter_parameters will be a tuple with the first index
        # being the min upvotes, second index being a list of
        # invalid words.

        if not load_compressed_tally_dict:
            if load_tally_dict:  # Checks if user wants to load an existing tally_dict
                print("Beginning to load tally dictionary")
                tally_dict = pickle.load(open("tally_dict.pickle", "rb"))

            else:
                print("Beginning to create tally dictionary")
                tally_dict = self.count_tokens(file_paths, filter_parameters, track_progress)

        if load_compressed_tally_dict:  # Checks if user wants to load an existing compressed_dict
            print("Beginning to load compressed tally dictionary")
            compressed_tally_dict = pickle.load(open("compressed_tally_dict.pickle", "rb"))

        else:
            print("Beginning to create compressed tally dictionary")
            compressed_tally_dict = self.compress_vocab(tally_dict, pre_bpe_vocab_size, track_progress)

        print("Splitting tokens for BPE")
        tally_dict_ready_for_bpe = self.split_keys_for_bpe(compressed_tally_dict)

        # Counts the initial number of tokens
        num_of_tokens = self.count_bpe_tokens(tally_dict_ready_for_bpe) + 1
        # Finds the most common pair of tokens and merges them
        final_dict = self.byte_pair_encodings_creator(tally_dict_ready_for_bpe, track_progress)

        while num_of_tokens <= vocab_size:
            # Keeps finding the most common pair and merging them until the wanted size is reached
            final_dict = self.byte_pair_encodings_creator(final_dict, track_progress)
            num_of_tokens += 1

            if track_progress > 0 and (num_of_tokens % 25) == 0:
                print(f"Number of tokens: {num_of_tokens}")

        # takes the jumble of split tokens and finds all the unique tokens and gives them a
        # unique id in a dictionary
        final_dict = self.convert_to_vocab_dict_with_ids(final_dict)

        # Saves the vocab dictionary
        pickle.dump(final_dict, open("vocab_dict.pickle", "wb"))
        return final_dict

    def convert_to_vocab_dict_with_ids(self, vocab_dict):
        # takes the jumble of split tokens and finds all the unique tokens and gives them a
        # unique id in a dictionary
        vocab_list = ["<|GEN|>", "<|ENDGEN|>", "<|BOS|>", "<|EOS|>"]
        vocab_list += self.count_bpe_tokens(vocab_dict, return_list=True)  # Returns a list of unique tokens
        vocab_dict = {}
        for count, token in enumerate(vocab_list):  # Puts unique tokens into the dictionary with unique ids
            vocab_dict[token] = count
        return vocab_dict

    @staticmethod
    def count_bpe_tokens(tally_dict, return_list=False):

        tokens = []
        for key, value in tally_dict.items():  # Counts all unique tokens from tally_dict
            for key_token in key:
                if key_token not in tokens:
                    tokens.append(key_token)
        if not return_list:
            return len(tokens)  # Returns how many unique tokens there are
        else:
            return tokens  # Returns all unique tokens in a list

    def byte_pair_encodings_creator(self, tally_dict, track_progress):

        text_category = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890'-"
        byte_counter = {}
        for key, value in tally_dict.items():  # Goes through dict and tallies most common pairs of tokens
            for i in range(len(key) - 1):
                if (len(key[i]) > 1 or key[i] in text_category) and (len(key[i+1]) > 1 or key[i+1] in text_category):
                    # makes sure it is not merging punctuation and words
                    # (excluding ‘ and - because we often want these to merge), so we don't end up with lots
                    # of different variations of the same word: play play, play! play? etc.
                    if f"{key[i]}{key[i + 1]}" not in byte_counter:  # If it is an unseen pair it will add
                        byte_counter[f"{key[i]}{key[i + 1]}"] = value  # the pair with how many occurrences there are
                    else:
                        byte_counter[f"{key[i]}{key[i + 1]}"] = byte_counter[f"{key[i]}{key[i + 1]}"] + value
                        # if it is already in the dict, it will just add the new occurrences

        largest_pair = 0
        most_common_pair = ""
        for key, value in byte_counter.items():  # Finds most common pair of tokens
            if value > largest_pair:
                largest_pair = value
                most_common_pair = key

        if track_progress > 0:
            print(f"Most common pair of tokens: {most_common_pair}")

        to_remove_from_dict = []
        to_add_to_dict = []
        for key, value in tally_dict.items():  # Edits the dict to put most common pair of tokens together
            new_key = self.edit_key(key, most_common_pair)
            if new_key is not None:
                to_remove_from_dict.append(key)
                to_add_to_dict.append((tuple(new_key), value))
                # print(new_key)

        for i in to_remove_from_dict:
            tally_dict.pop(i)
        for key, value in to_add_to_dict:
            tally_dict[key] = value

        return tally_dict

    @staticmethod
    def split_keys_for_bpe(tally_dict):
        """
        Creates a new dict with every token split into its chars,
        so it is ready for use to create the byte pair encodings.
        """
        new_tally_dict = {}

        for key, value in tally_dict.items():
            key_list = list(key)
            # key_list.append("</w>")
            new_tally_dict[tuple(key_list)] = value
        return new_tally_dict

    def edit_key(self, key, most_common_token):
        for i in range(len(key) - 1):
            if f"{key[i]}{key[i + 1]}" == most_common_token:
                # Will check for most_common_token in each key and merge them into one.
                key_list = list(key)
                key_list[i] = most_common_token
                del key_list[i + 1]
                # print(key_list)
                re_check = self.edit_key(key_list, most_common_token)
                # It then checks if there are any other occurrences of the most_common_token
                # in a recursive function, which will keep replacing all of most_common_token
                # until all have been replaced
                if re_check is not None:
                    return re_check
                else:
                    return key_list
        return None

    @staticmethod
    def compress_vocab(tally_dict, pre_bpe_vocab_size, track_progress):
        # Shrinks the dictionary, removing the rarest tokens first until it reaches wanted vocab size

        to_remove = []
        minimum_occurrences = 1
        while len(tally_dict) >= pre_bpe_vocab_size:  # Keeps shrinking it until it is at the correct length
            for key, value in tally_dict.items():
                if value <= minimum_occurrences:  # adds key to a removal list if it doesn't occur enough times.
                    to_remove.append(key)

            for key_to_remove in to_remove:  # removes the keys that don't occur enough times.
                tally_dict.pop(key_to_remove)

            minimum_occurrences += 1  # Increases minimum occurrences that the token must come up
            to_remove = []
            if track_progress > 0:
                print(f"Current length of dict: {len(tally_dict)}")

        # pickle.dump(tally_dict, open("compressed_tally_dict.pickle", "wb"))  # Saves the dict after
        return tally_dict

    @staticmethod
    def count_tokens(file_paths, filter_parameters=(2, ["deleted", "removed"]),
                     track_progress=200000):
        # Goes through each file and counts each individual token
        # punctuation = '!"£$%^&*()_+={[}]:;@~#<,>.?/'

        counting_dict = {}
        start_time = time.time()

        for file_path in file_paths:  # iterates through each file
            json_file = open(file_path, "r")  # reads the current file

            if track_progress > 0:
                print(f"Processing: {file_path}")

            for line_num, line in enumerate(json_file):  # iterates over each line of the file

                if (track_progress > 0) and ((line_num % track_progress) == 0) and line_num != 0:
                    print(f"\nTime elapsed: {round(time.time() - start_time, 2)} seconds.")
                    print(f"Lines processed: {line_num}")

                current_line_converted = json.loads(line)  # converts the json into a python tuple

                if int(current_line_converted["score"]) >= filter_parameters[0]:
                    # checks if score is greater than given amount

                    body = ""
                    if "RC" in file_path:  # Checks if it is a comment
                        body = current_line_converted["body"]  # gets the text from the body of comment

                    elif "RS" in file_path:  # Checks if it is a submission
                        self_text = current_line_converted["selftext"]
                        title = current_line_converted["title"]
                        body = f"{title} {self_text}"
                        # gets the text from the self_text and title of submission

                    flag = False
                    for banned_token in filter_parameters[1]:
                        # Validation: makes sure it doesn't contain banned tokens such as deleted or removed.
                        if banned_token in body:
                            flag = True

                    if not flag:  # continues if it doesn't include banned tokens
                        body = body.split()  # splits the body up
                        for token in body:  # iterates through all tokens
                            if token not in counting_dict:  # checks if token is not in dictionary
                                counting_dict[token] = 1  # adds token to counting_dict

                            else:
                                counting_dict[token] = counting_dict[token] + 1  # adds 1 to the occurrence of the token

            pickle.dump(counting_dict, open("tally_dict.pickle", "wb"))  # saves the tally_dict after every file
        return counting_dict

