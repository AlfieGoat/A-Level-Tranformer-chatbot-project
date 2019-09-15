import pre_processing_raw_training_data_database
import pre_processing_database
from pre_processing_convert_to_bpes import convert_tokens_to_bpes
import pickle
import torch
import time


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


shelf_db = pre_processing_raw_training_data_database.ShelfDB()
comment_db = pre_processing_database.Database()

vocab = pickle.load(open("vocab_dict.pickle", "rb"))
iterable_comments = comment_db.iterate_over_comments()

vocab_list = []
for key, value in vocab.items():
    vocab_list.append(key)

count = -1
start = time.time()
for index, comment_data in enumerate(iterable_comments):
    """
    indexing from database
    0: id
    1: parent_id
    2: score
    3: subreddit
    4: comment
    """
    if index >= 0:
        parent = get_parent(comment_data[1], comment_db)
        if parent is not None:
            tokenised_comment_tensor = create_torch_tensor(comment_data[4], gen=True)
            if tokenised_comment_tensor is not None:
                count += 1
                shelf_db.add_rows(count, (tokenised_comment_tensor, parent))
                # print(tokenised_comment_tensor, parent)
                if count % 1000 == 0:
                    print(f"Processed: {count} \t\t\t\t\tTime elapsed: {round(time.time() - start , 2)}")
                    with open("current.txt", "w") as file:
                        file.write(str(index))







