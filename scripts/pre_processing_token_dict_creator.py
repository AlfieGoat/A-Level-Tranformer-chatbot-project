import json
import time

class TokenDictCreator:

    def create_dict(self, file_paths, token_dict_save_path,
                    vocab_size, filter_parameters=(2, ["deleted", "removed"]),
                    track_progress=50000, batch_size=10000000):
        # track_progress is the num of rows to process before
        # giving elapsed time. Set to 0 for no tracking
        # filter_parameters will be a tuple with the first index
        # being the min upvotes, second index being list of invalid
        # words

        tally_dict = self.count_tokens(file_paths, filter_parameters, track_progress, batch_size)

    @staticmethod
    def count_tokens(file_paths, filter_parameters,
                     track_progress=50000, batch_size=10000000):

        punctuation = '!"£$%^&*()_+={[}]:;@~#<,>.?/'
        batch = []
        counting_dict = {}
        if track_progress > 0:
            start_time = time.time()

        for file_path in file_paths:
            json_file = open(file_path, "r")

            if track_progress > 0:
                print(f"Processing: {file_path}")

            for line_num, line in enumerate(json_file):
                batch.append(line)

                if (track_progress > 0) and ((line_num % track_progress) == 0):
                    print(f"Time elapsed: {round(start_time - time.time(), 2)} seconds.")

                if (line_num % batch_size) == 0 and line_num != 0:

                    for batch_line in batch:
                        current_line_converted = json.loads(batch_line)

                        if int(current_line_converted["score"]) >= filter_parameters[0]:
                            body = current_line_converted["body"]

                            for i in punctuation:
                                body = body.replace(i, f" {i} ")

                            body = body.split()

                            for banned_tokens in filter_parameters[1]:
                                if banned_tokens in current_line_converted:
                                    break

                            for token in body:

                                if token not in counting_dict:
                                    counting_dict[token] = 0

                                else:
                                    counting_dict[token] = counting_dict[token] + 1
                    batch = []

        return counting_dict

    # TODO def save_dict(self):



































