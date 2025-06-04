import os
from dotenv import load_dotenv
from tag.database.db_connection import connect_db
from tag.database.schema_loader import load_schema
from tag.src.text2sqlchain import generate_sql
from tag.src.query_executor import execute_text2sql_response
from tag.src.answer_generator import generate_answer


from langchain_core.runnables import RunnableLambda

# Load env vars
load_dotenv()

# Init DB
conn = connect_db()
schema = load_schema(conn)


def tag_pipeline(input_dict):
    question = input_dict["question"]

    sql = generate_sql(schema, question, top_k=100)
    rows, columns = execute_text2sql_response(conn, sql)
    answer = generate_answer(columns, rows, question)
    table = format_table(columns, rows)

    return {
        "sql": sql,
        "table_result": table,
        "answer": answer
    }


def format_table(columns, rows):
    if not rows:
        return "⚠️ Tidak ada hasil ditemukan."

    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = "\n".join(["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows])
    return "\n".join([header, separator, body])

# LangChain Runnable
qa_chain = RunnableLambda(tag_pipeline)

