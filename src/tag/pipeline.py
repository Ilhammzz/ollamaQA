from langchain_core.runnables import RunnableLambda
from src.text2sqlchain import generate_sql
from src.query_executor import execute_text2sql_response
from src.answer_generator import generate_answer
from tag.database.db_connection import connect_db
from tag.database.schema_loader import load_schema

# Inisialisasi koneksi dan schema di awal (bisa di-cache)
conn = connect_db()
schema = load_schema(conn)

# Fungsi utama pipeline yang dibungkus ke Runnable
def tag_pipeline(input_dict):
    question = input_dict["question"]
    mode = input_dict.get("mode", "zero-shot")  # Default mode adalah zero-shot

    # Step 1: Generate SQL dari pertanyaan
    sql = generate_sql(schema, question, top_k=100, mode=mode)

    # Step 2: Eksekusi SQL
    rows, columns = execute_text2sql_response(conn, sql)

    # Step 3: Buat jawaban natural language
    answer = generate_answer(columns, rows, question, mode=mode)

    list_of_strings = []
    for row in rows:
        row_dict = {columns[i]: row[i] for i in range(len(columns))}
        string_dict = str(row_dict)
        list_of_strings.append(string_dict)

    return {
        "sql": sql,
        "columns": columns,
        "table_result": list_of_strings,
        "answer": answer
    }
# Bungkus dalam LangChain Runnable
qa_chain = RunnableLambda(tag_pipeline)
