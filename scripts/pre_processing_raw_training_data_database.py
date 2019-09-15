import shelve


class ShelfDB:

    def __init__(self):
        self.db = shelve.open("database_shelf")

    def add_rows(self, key, value):
        self.db[str(key)] = value

    def retrieve_row(self, key):
        return self.db[str(key)]
