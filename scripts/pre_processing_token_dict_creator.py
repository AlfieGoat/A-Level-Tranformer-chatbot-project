import json
import time
import pickle


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

        if load_tally_dict and not load_compressed_tally_dict:
            print("Beginning to load tally dictionary")
            tally_dict = pickle.load(open("tally_dict.pickle", "rb"))

        else:
            print("Beginning to create tally dictionary")
            tally_dict = self.count_tokens(file_paths, filter_parameters, track_progress)

        if load_compressed_tally_dict:
            print("Beginning to load compressed tally dictionary")
            compressed_tally_dict = pickle.load(open("compressed_tally_dict.pickle", "rb"))
        else:
            print("Beginning to create compressed tally dictionary")
            compressed_tally_dict = self.compress_vocab(tally_dict, pre_bpe_vocab_size, track_progress)

        print("Splitting tokens and adding </w> to the end for BPE")
        tally_dict_ready_for_bpe = self.split_keys_for_bpe(compressed_tally_dict)

        num_of_tokens = self.count_bpe_tokens(tally_dict_ready_for_bpe) + 1
        final_dict = self.byte_pair_encodings_creator(tally_dict_ready_for_bpe, track_progress)

        while num_of_tokens <= vocab_size:
            final_dict = self.byte_pair_encodings_creator(final_dict, track_progress)
            num_of_tokens += 1

            if track_progress > 0 and (num_of_tokens % 25) == 0:
                print(f"Number of tokens: {num_of_tokens}")

        pickle.dump(final_dict, open("vocab_dict.pickle", "wb"))
        return final_dict

    @staticmethod
    def count_bpe_tokens(tally_dict, return_list=False):

        token_counter = []
        for key, value in tally_dict.items():
            for key_token in key:
                if key_token not in token_counter:
                    token_counter.append(key_token)
        if not return_list:
            return len(token_counter)
        else:
            return token_counter

    def byte_pair_encodings_creator(self, tally_dict, track_progress):

        text_category = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890'-"
        byte_counter = {}
        for key, value in tally_dict.items():  # Goes through dict and tallies most common pairs of tokens
            for i in range(len(key) - 1):
                if (len(key[i]) > 1 or key[i] in text_category) and (len(key[i+1]) > 1 or key[i+1] in text_category):
                    if f"{key[i]}{key[i + 1]}" not in byte_counter:
                        byte_counter[f"{key[i]}{key[i + 1]}"] = value
                    else:
                        byte_counter[f"{key[i]}{key[i + 1]}"] = byte_counter[f"{key[i]}{key[i + 1]}"] + value

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
        Creates a new dict with every token split into its chars with a </w> on the
        end, so it is ready for use to create the byte pair encodings.
        """
        new_tally_dict = {}

        for key, value in tally_dict.items():
            key_list = list(key)
            key_list.append("</w>")
            new_tally_dict[tuple(key_list)] = value
        return new_tally_dict

    def edit_key(self, key, most_common_token):
        for i in range(len(key) - 1):
            if f"{key[i]}{key[i + 1]}" == most_common_token:

                key_list = list(key)
                key_list[i] = most_common_token
                del key_list[i + 1]
                # print(key_list)
                re_check = self.edit_key(key_list, most_common_token)
                if re_check is not None:
                    return re_check
                else:
                    return key_list
        return None

    @staticmethod
    def compress_vocab(tally_dict, pre_bpe_vocab_size, track_progress):
        # Shrinks the dictionary, removing the rarest tokens first until it reaches wanted vocab size
        to_remove = []
        key_delete = 1
        while len(tally_dict) >= pre_bpe_vocab_size:
            for key, value in tally_dict.items():
                if value == key_delete:
                    to_remove.append(key)

            for key_to_remove in to_remove:
                tally_dict.pop(key_to_remove)

            key_delete += 1
            to_remove = []
            if track_progress > 0:
                print(f"Current length of dict: {len(tally_dict)}")

        pickle.dump(tally_dict, open("compressed_tally_dict.pickle", "wb"))
        return tally_dict

    @staticmethod
    def count_tokens(file_paths, filter_parameters=(2, ["deleted", "removed"]),
                     track_progress=200000):

        # punctuation = '!"£$%^&*()_+={[}]:;@~#<,>.?/'
        counting_dict = {}
        # counting_dict = pickle.load(open("tally_dict.pickle", "rb"))

        start_time = time.time()

        for file_path in file_paths:
            json_file = open(file_path, "r")

            if track_progress > 0:
                print(f"Processing: {file_path}")

            for line_num, line in enumerate(json_file):

                if (track_progress > 0) and ((line_num % track_progress) == 0) and line_num != 0:
                    print(f"\nTime elapsed: {round(time.time() - start_time, 2)} seconds.")
                    print(f"Lines processed: {line_num}")

                current_line_converted = json.loads(line)

                if int(current_line_converted["score"]) >= filter_parameters[0]:
                    if "RC" in file_path:
                        body = current_line_converted["body"]

                    else:
                        self_text = current_line_converted["selftext"]
                        title = current_line_converted["title"]
                        body = f"{title} {self_text}"

                    body = body.split()

                    flag = False
                    for banned_token in filter_parameters[1]:
                        if banned_token in body:
                            flag = True

                    if not flag:
                        for token in body:

                            if token not in counting_dict:
                                counting_dict[token] = 1

                            else:
                                counting_dict[token] = counting_dict[token] + 1
            pickle.dump(counting_dict, open("tally_dict.pickle", "wb"))
        return counting_dict
