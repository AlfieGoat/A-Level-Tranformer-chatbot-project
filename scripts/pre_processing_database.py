import sqlite3


class Database:

    def __init__(self):
        self.db = sqlite3.connect("database")  # Connects to db file, or creates it if it doesn't exist
        self.cursor = self.db.cursor()
        self.cursor2 = self.db.cursor()
        # Creates the tables if the don't exist with the required fields
        self.cursor.execute("CREATE TABLE IF NOT EXISTS "
                            "comments"
                            "(id TEXT PRIMARY KEY, "
                            "parent_id TEXT, "
                            "score INT, subreddit TEXT, "
                            "text TEXT)")
        self.cursor.execute("CREATE TABLE IF NOT EXISTS "
                            "submissions"
                            "(id TEXT PRIMARY KEY, "
                            "score INT, subreddit TEXT, "
                            "text TEXT)")
        self.db.commit()

    # This is a way of adding many comments to the db
    def add_comments_to_db(self, cache):
        self.db.executemany("INSERT INTO COMMENTS(id, parent_id, score, "
                            "subreddit, text) VALUES(?, ?, ?, ?, ?)", cache)
        self.db.commit()

    # This is a way of adding many submissions to the db
    def add_submissions_to_db(self, cache):
        self.db.executemany("INSERT INTO SUBMISSIONS (id, score, "
                            "subreddit, text) VALUES(?, ?, ?, ?)", cache)
        self.db.commit()

    # A way of iterating over all the comments
    def iterate_over_submissions(self):
        return self.cursor.execute('SELECT * FROM SUBMISSIONS')

    # A way of iterating over all the submissions
    def iterate_over_comments(self):
        return self.cursor.execute('SELECT * FROM COMMENTS')

    # A way of getting a comment with a specific id
    def get_comment_by_id(self, comment_id):
        return self.cursor2.execute(f"SELECT * FROM COMMENTS WHERE id ='{comment_id}'").fetchone()

    # A way of getting a submission with a specific id
    def get_submission_by_id(self, submission_id):
        return self.cursor2.execute(f"SELECT * FROM SUBMISSIONS WHERE id ='{submission_id}'").fetchone()

    def custom_call(self, call):
        return self.cursor.execute(call)




