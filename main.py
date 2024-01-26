# Created by trilo at 15-01-2024
from question_answer_worker import palm_llm, query, huggingface_embedding, croma_vectorstore
from database_fun import db_connection
from prompt_engg import few_shot_learning, few_shot_prompt, prompt

def main():
    # calling Palm LLM
    llm = palm_llm()

    # Database Connection
    db, db_chain = db_connection(llm)

    # calling query
    qns1, qns2, qns3, qns4, qns5 = query(db_chain)

    # Few shot learning
    few_shots = few_shot_learning(qns1, qns2, qns3, qns4, qns5)

    # Huggingface embedding
    embeddings, to_vectorize = huggingface_embedding(few_shots)

    # Croma for vector Store
    example_selector = croma_vectorstore(to_vectorize, embeddings, few_shots)

    # Prompt Engg
    mysql_prompt = prompt()

    # Few Shot Prompt
    new_chain = few_shot_prompt(example_selector, mysql_prompt, db, llm)

    # print(new_chain("How much is the price of the inventory for all extra large size t-shirts?"))
    return new_chain