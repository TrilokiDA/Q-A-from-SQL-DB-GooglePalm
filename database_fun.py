# Created by trilo at 19-01-2024

from langchain.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain

##### Database credetial and connection #####
def db_connection(llm):
    db_user = "root"
    db_password = "root"
    db_host = "localhost"
    db_name = "atliq_tshirts"
    db = SQLDatabase.from_uri(f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}",
                              sample_rows_in_table_info=3)
    db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True)
    return db, db_chain
