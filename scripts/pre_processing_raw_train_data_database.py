import sqlite3


class Database:

    def __init__(self):
        self.db = sqlite3.connect("C:/training_db")  # Connects to db file, or creates it if it doesn't exist
        #self.db = sqlite3.connect("taining_data_db")  # Connects to db file, or creates it if it doesn't exist
        self.cursor = self.db.cursor()
        self.cursor2 = self.db.cursor()
        # Creates the tables if the don't exist with the required fields
        self.cursor.execute("CREATE TABLE IF NOT EXISTS "
                            "trainData"
                            "(id INT PRIMARY KEY, "
                            "child BLOB, "
                            "parent BLOB, "
                            "parent_length,"
                            "child_length)")

        self.db.commit()

    # This is a way of adding many comments to the db
    def add_train_data_to_db(self, cache):
        self.db.executemany("REPLACE INTO trainData(id, child, parent) "
                            "VALUES(?, ?, ?)", cache)
        self.db.commit()

    # A way of iterating over all the comments
    def iterate_over_train_data(self):
        return self.cursor.execute('SELECT * FROM trainData')

    # A way of getting a comment with a specific id
    def get_train_data_by_id(self, comment_id):
        return self.cursor2.execute(f"SELECT * FROM trainData WHERE id ='{comment_id}'").fetchone()

    def add_train_data_to_sorted_db(self, cache, child_length):
        self.cursor.execute(f"CREATE TABLE IF NOT EXISTS "
                            f"child{child_length}"
                            "(id INT PRIMARY KEY, "
                            "child BLOB, "
                            "parent BLOB)")
        self.db.executemany(f"REPLACE INTO child{child_length}(id, child, parent) "
                            "VALUES(?, ?, ?)", cache)
        self.db.commit()

    def custom_call(self, call):
        return self.cursor.execute(call)

    def update(self, call, data):
        self.cursor2.executemany(call, data)
        self.db.commit()
