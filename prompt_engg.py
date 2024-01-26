# Created by trilo at 20-01-2024
from langchain.prompts import FewShotPromptTemplate
from langchain.chains.sql_database.prompt import PROMPT_SUFFIX, _mysql_prompt
from langchain.prompts.prompt import PromptTemplate
from langchain_experimental.sql import SQLDatabaseChain

def few_shot_learning(qns1, qns2, qns3, qns4, qns5):
    few_shots = [
        {'Question': "How many t-shirts do we have left for levis in extra small size and white color?",
         'SQLQuery': "SELECT sum(stock_quantity) FROM t_shirts WHERE brand = 'Nike' AND color = 'White' AND size = 'XS'",
         'SQLResult': "Result of the SQL query",
         'Answer': qns1},
        {'Question': "How much is the total price of the inventory for all S-size t-shirts?",
         'SQLQuery': "SELECT SUM(price*stock_quantity) FROM t_shirts WHERE size = 'S'",
         'SQLResult': "Result of the SQL query",
         'Answer': qns2},
        {'Question': "If we have to sell all the Levi’s T-shirts today with discounts applied. How much revenue  our "
                     "store will generate (post discounts)?",
         'SQLQuery': "SELECT sum(a.total_amount * ((100-COALESCE(discounts.pct_discount,0))/100)) as total_revenue from"
                     " (select sum(price*stock_quantity) as total_amount, t_shirt_id from t_shirts where brand = "
                     "'Levi' group by t_shirt_id) a left join discounts on a.t_shirt_id = discounts.t_shirt_id",
         'SQLResult': "Result of the SQL query",
         'Answer': qns3},
        {'Question': "If we have to sell all the Levi’s T-shirts today. How much revenue our store will generate "
                     "without discount?",
         'SQLQuery': "SELECT SUM(price * stock_quantity) FROM t_shirts WHERE brand = 'Levi'",
         'SQLResult': "Result of the SQL query",
         'Answer': qns4},
        {'Question': "How many white color Levi's t-shirts we have available?",
         'SQLQuery': "SELECT sum(stock_quantity) FROM t_shirts WHERE brand = 'Levi' AND color = 'White'",
         'SQLResult': "Result of the SQL query",
         'Answer': qns5
         }
    ]
    return few_shots

def prompt():
    ### my sql based instruction prompt
    mysql_prompt = """You are a MySQL expert. Given an input question, first create a syntactically correct MySQL 
    query to run, then look at the results of the query and return the answer to the input question. Unless the user 
    specifies in the question a specific number of examples to obtain, query for at most {top_k} results using the 
    LIMIT clause as per MySQL. You can order the results to return the most informative data in the database. Never 
    query for all columns from a table. You must query only the columns that are needed to answer the question. Wrap 
    each column name in backticks (`) to denote them as delimited identifiers. Pay attention to use only the column 
    names you can see in the tables below. Be careful to not query for columns that do not exist. Also, pay attention 
    to which column is in which table. Pay attention to use CURDATE() function to get the current date, 
    if the question involves "today".
    
    Use the following format:
    
    Question: Question here
    SQLQuery: Query to run with no pre-amble
    SQLResult: Result of the SQLQuery
    Answer: Final answer here
    
    No pre-amble.
    """
    return mysql_prompt

def few_shot_prompt(example_selector, mysql_prompt, db, llm):
    example_prompt = PromptTemplate(
        input_variables=["Question", "SQLQuery", "SQLResult", "Answer", ],
        template="\nQuestion: {Question}\nSQLQuery: {SQLQuery}\nSQLResult: {SQLResult}\nAnswer: {Answer}",
    )
    few_shot_prompt = FewShotPromptTemplate(
        example_selector=example_selector,
        example_prompt=example_prompt,
        prefix=mysql_prompt,
        suffix=PROMPT_SUFFIX,
        input_variables=["input", "table_info", "top_k"],  # These variables are used in the prefix and suffix
    )
    new_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True, prompt=few_shot_prompt)
    return new_chain