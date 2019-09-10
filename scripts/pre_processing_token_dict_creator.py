import json
import time
import pickle


class TokenDictCreator:

    def create_dict(self, file_paths, token_dict_save_path,
                    vocab_size, filter_parameters=(2, ["deleted", "removed"]),
                    track_progress=2000000):
        # track_progress is the num of rows to process before
        # giving elapsed time. Set to 0 for no tracking
        # filter_parameters will be a tuple with the first index
        # being the min upvotes, second index being list of invalid
        # words

        tally_dict = self.count_tokens(file_paths, filter_parameters, track_progress)

    @staticmethod
    def count_tokens(file_paths, filter_parameters=(2, ["deleted", "removed"]),
                     track_progress=200000):

        punctuation = '!"£$%^&*()_+={[}]:;@~#<,>.?/'
        counting_dict = {}
        if track_progress > 0:
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
                    body = current_line_converted["body"]

                    for i in punctuation:
                        body = body.replace(i, f" {i} ")

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


        return counting_dict

    # TODO def save_dict(self):



































