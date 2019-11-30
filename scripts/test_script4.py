import pre_processing_raw_train_data_database

db = pre_processing_raw_train_data_database.Database()

sqlQuery = "Select * from SQLite_master"


print(db.custom_call(sqlQuery).fetchall())


#db.custom_call("CREATE INDEX parent_length_index ON trainData (parent_length)")
#db.custom_call("CREATE INDEX child_length_index ON trainData (child_length)")
#db.custom_call("CREATE INDEX parent_child_length_index ON trainData (parent_length, child_length)")
#db.custom_call("CREATE INDEX child_parent_length_index ON trainData (child_length, parent_length)")
"""
iterable = db.custom_call("SELECT * FROM trainData ORDER BY parent_length, child_length DESC")
for i in iterable:
    print(i[0], i[3:])
"""
