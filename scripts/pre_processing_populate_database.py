from pre_processing_database import Database
import time
import json

# This Script adds all submissions and comments to the database
# from a list of given file paths. It first caches the comments
# then once it has reached a certain size, it will push those to
# the correct table. It will also make sure it doesn't include
# deleted or removed comments/submissions.

file_paths = ["Z:/Code/A-Level-Tranformer-chatbot-project/data/uncompressed/RS_2019-05",
              "Z:/Code/A-Level-Tranformer-chatbot-project/data/uncompressed/RC_2019-05",
              "Z:/Code/A-Level-Tranformer-chatbot-project/data/uncompressed/RC_2019-04",
              "Z:/Code/A-Level-Tranformer-chatbot-project/data/uncompressed/RC_2019-03",
              "Z:/Code/A-Level-Tranformer-chatbot-project/data/uncompressed/RC_2019-02",
              "Z:/Code/A-Level-Tranformer-chatbot-project/data/uncompressed/RC_2019-01",
              "Z:/Code/A-Level-Tranformer-chatbot-project/data/uncompressed/RC_2018-12",
              "Z:/Code/A-Level-Tranformer-chatbot-project/data/uncompressed/RC_2018-11",
              "Z:/Code/A-Level-Tranformer-chatbot-project/data/uncompressed/RC_2018-10",
              "Z:/Code/A-Level-Tranformer-chatbot-project/data/uncompressed/RC_2018-09",
              "Z:/Code/A-Level-Tranformer-chatbot-project/data/uncompressed/RS_2018-08",
              "Z:/Code/A-Level-Tranformer-chatbot-project/data/uncompressed/RS_2018-09",
              "Z:/Code/A-Level-Tranformer-chatbot-project/data/uncompressed/RS_2018-10",
              "Z:/Code/A-Level-Tranformer-chatbot-project/data/uncompressed/RS_2018-11",
              "Z:/Code/A-Level-Tranformer-chatbot-project/data/uncompressed/RS_2018-12",
              "Z:/Code/A-Level-Tranformer-chatbot-project/data/uncompressed/RS_2019-01",
              "Z:/Code/A-Level-Tranformer-chatbot-project/data/uncompressed/RS_2019-02",
              "Z:/Code/A-Level-Tranformer-chatbot-project/data/uncompressed/RS_2019-03",
              "Z:/Code/A-Level-Tranformer-chatbot-project/data/uncompressed/RS_2019-04"]

db = Database()
cache_comments = []
cache_submissions = []
start_time = time.time()

for file_name in file_paths:  # iterates through the files
    print(f"Beginning to process: {file_name}")
    json_file = open(file_name, "r")  # Opens the file

    for line_num, line in enumerate(json_file):  # Iterates through the file

        if ((line_num % 500000) == 0) and line_num != 0:  # If it has processed n number of lines, it will push them
            print(f"\nTime elapsed: {round(time.time() - start_time, 2)} seconds.")
            print(f"Lines processed: {line_num}")
            db.add_comments_to_db(cache_comments)
            db.add_submissions_to_db(cache_submissions)
            # Empties the cache
            cache_comments = []
            cache_submissions = []

        line_converted = json.loads(line)  # Loads the line of the file from json into a python tuple
        if line_converted["score"] > 1:  # Checks that the score of the comment is at least 1
            if "RC" in file_name:  # Checks if it is a comment
                if "deleted" not in line_converted["body"] and "removed" not in line_converted["body"]:
                    # Checks if comment has been deleted or removed
                    cache_comments.append((f"t1_{line_converted['id']}",
                                           line_converted["parent_id"],
                                           line_converted["score"],
                                           line_converted["subreddit"],
                                           line_converted["body"]))
                    # Adds the comment to the cache, ready to be pushed to the db.

            elif "RS" in file_name:  # Checks if it is a submission
                if "deleted" not in line_converted["title"] and "removed" not in line_converted["title"]:
                    if "deleted" not in line_converted["selftext"] and "removed" not in line_converted["selftext"]:
                        # Checks if submission has been deleted or removed
                        cache_submissions.append((f"t3_{line_converted['id']}",
                                                  line_converted["score"],
                                                  line_converted["subreddit"],
                                                  f"{line_converted['title']}. {line_converted['selftext']}"))
                        # Adds the submission to the cache, ready to be pushed to the db.
