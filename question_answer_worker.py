# Created by trilo at 19-01-2024
from langchain.llms import GooglePalm
from langchain_experimental.sql import SQLDatabaseChain
from langchain.prompts import SemanticSimilarityExampleSelector
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

def palm_llm():
    ##### Google LLM api key and connection #####
    api_key = 'YOUR_API_KEY'
    llm = GooglePalm(google_api_key=api_key, temperature=0.2)
    return llm

def query(db_chain):
    #### For more detail please refer "testing query" section in Jupyter file ####
    qns1 = db_chain.run("How many t-shirts do we have left for levis in extra small size and white color?")
    qns2 = db_chain.run("SELECT SUM(price*stock_quantity) FROM t_shirts WHERE size = 'S';")
    sql_code = """
    select sum(a.total_amount * ((100-COALESCE(discounts.pct_discount,0))/100)) as total_revenue from
    (select sum(price*stock_quantity) as total_amount, t_shirt_id from t_shirts where brand = 'Levi'
    group by t_shirt_id) a left join discounts on a.t_shirt_id = discounts.t_shirt_id
     """
    qns3 = db_chain.run(sql_code)
    qns4 = db_chain.run("SELECT SUM(price * stock_quantity) FROM t_shirts WHERE brand = 'Levi'")
    qns5 = db_chain.run("SELECT sum(stock_quantity) FROM t_shirts WHERE brand = 'Levi' AND color = 'White'")
    return qns1, qns2, qns3, qns4, qns5

def huggingface_embedding(few_shots):
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    to_vectorize = [" ".join(example.values()) for example in few_shots]
    return embeddings, to_vectorize

def croma_vectorstore(to_vectorize, embeddings, few_shots):
    vectorstore = Chroma.from_texts(to_vectorize, embeddings, metadatas=few_shots)
    example_selector = SemanticSimilarityExampleSelector(vectorstore=vectorstore, k=2, )
    return example_selector
