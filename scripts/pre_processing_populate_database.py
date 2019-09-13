from pre_processing_database import Database
import time
import json

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
line = 1
start_time = time.time()

for file_name in file_paths:
    print(f"Beginning to process: {file_name}")
    json_file = open(file_name, "r")

    for line_num, line in enumerate(json_file):

        if ((line_num % 500000) == 0) and line_num != 0:
            print(f"\nTime elapsed: {round(time.time() - start_time, 2)} seconds.")
            print(f"Lines processed: {line_num}")
            db.add_comments_to_db(cache_comments)
            db.add_submissions_to_db(cache_submissions)
            cache_comments = []
            cache_submissions = []

        line_converted = json.loads(line)
        if line_converted["score"] > 1:
            if "RC" in file_name:
                if "deleted" not in line_converted["body"] and "removed" not in line_converted["body"]:
                    cache_comments.append((f"t1_{line_converted['id']}",
                                           line_converted["parent_id"],
                                           line_converted["score"],
                                           line_converted["subreddit"],
                                           line_converted["body"]))

            elif "RS" in file_name:
                if "deleted" not in line_converted["title"] and "removed" not in line_converted["title"]:
                    if "deleted" not in line_converted["selftext"] and "removed" not in line_converted["selftext"]:
                        cache_submissions.append((f"t3_{line_converted['id']}",
                                                  line_converted["score"],
                                                  line_converted["subreddit"],
                                                  f"{line_converted['title']}. {line_converted['selftext']}"))
