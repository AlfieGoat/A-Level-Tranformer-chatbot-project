import sqlite3


class Database:

    def __init__(self):
        self.db = sqlite3.connect("database")
        self.cursor = self.db.cursor()
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

    def add_comments_to_db(self, cache):
        self.db.executemany("INSERT INTO COMMENTS(id, parent_id, score, "
                            "subreddit, text) VALUES(?, ?, ?, ?, ?)", cache)
        self.db.commit()

    def add_submissions_to_db(self, cache):
        self.db.executemany("INSERT INTO SUBMISSIONS (id, score, "
                            "subreddit, text) VALUES(?, ?, ?, ?)", cache)
        self.db.commit()

    def iterate_over_submissions(self):
        return self.cursor.execute('SELECT * FROM SUBMISSIONS')

    def iterate_over_comments(self):
        return self.cursor.execute('SELECT * FROM COMMENTS')







