import pre_processing_raw_training_data_database
import pre_processing_database
from pre_processing_convert_to_bpes import convert_tokens_to_bpes
import pickle
import torch
import time
import pre_processing_raw_train_data_database
import pickle

def get_parent(id, db):
    if id[1] == "1":
        # comment
        comment = db.get_comment_by_id(id)
        if comment is not None:
            tokenised_comment = create_torch_tensor(comment[4])
            if tokenised_comment is None:
                return None
            try:
                parent = get_parent(comment[1], db)
            except RecursionError:
                return None
            if parent is not None:
                return torch.cat((parent, tokenised_comment), dim=0)
            else:
                return tokenised_comment
        else:
            return None

    elif id[1] == "3":
        # Submission
        comment = db.get_submission_by_id(id)
        if comment is not None:
            tokenised_comment = create_torch_tensor(comment[3])
            return tokenised_comment
        else:
            return None


def create_torch_tensor(comment, gen=False):
    bpe_comment = convert_tokens_to_bpes(comment, vocab_list)
    if bpe_comment is None:
        return None
    tokenised_comment = torch.zeros(len(bpe_comment) + 2).int()
    if gen:
        tokenised_comment[0], tokenised_comment[-1] = 0, 1
    else:
        tokenised_comment[0], tokenised_comment[-1] = 2, 3
    # Sets the first token to be 0:<|GEN|> and the final to be 1:<|ENDGEN|>
    for index, bpe_token in enumerate(bpe_comment):
        tokenised_comment[index + 1] = vocab[bpe_token]

    return tokenised_comment


def main():
    count = 60054000  # ID number to pick up from, if I stop the script or it fails
    start = time.time()  # used for measuring time passed
    cache = []  # A cache which holds the comments temporarily before it is added to the database
    for index, comment_data in enumerate(iterable_comments):  # Iterates over all the comments
        """
        indexing from database
        0: id
        1: parent_id 
        2: score
        3: subreddit
        4: comment
        """
        if index >= 77565345:  # Index of database to continue getting comments from, if I stop the script or it fails
            parent = get_parent(comment_data[1], comment_db)  # Gets the parent comments
            if parent is not None:  # Makes sure parent comment exists
                # Converts the child comment into BPE tensor
                tokenised_comment_tensor = create_torch_tensor(comment_data[4], gen=True)
                # makes sure nothing went wrong whilst making tensor of child comment
                if tokenised_comment_tensor is not None:
                    parent = pickle.dumps(parent)  # Serialises the parent comment using pickle
                    # Serialises the child comment using pickle
                    tokenised_comment_tensor = pickle.dumps(tokenised_comment_tensor)
                    count += 1
                    # adds the serialised data to the cache with their count
                    cache.append((count, tokenised_comment_tensor, parent))
                    # print(tokenised_comment_tensor, parent)
                    if count % 1000 == 0:
                        # pushes the cache to DB every 1000 comments
                        print(f"Processed: {count} \t\t\t\t\tTime elapsed: {round(time.time() - start , 2)}")
                        db.add_train_data_to_db(cache)
                        cache = []  # Clears the cache
                        with open("current.txt", "w") as file:
                            file.write(str(index))  # updates file which keeps track of which index we are working on


if __name__ == "__main__":
    db = pre_processing_raw_train_data_database.Database()
    comment_db = pre_processing_database.Database()
    vocab = pickle.load(open("vocab_dict.pickle", "rb"))
    iterable_comments = comment_db.iterate_over_comments()

    vocab_list = []
    for key, value in vocab.items():
        vocab_list.append(key)
    main()






