def extract_sql_query_from_response(response: str) -> str:
    """
    Ekstrak SQL dari blok kode yang diawali ```sql dan diakhiri ```.
    """
    start_token = "```sql"
    end_token = "```"
    
    start_idx = response.find(start_token)
    if start_idx == -1:
        raise ValueError("Tidak ditemukan blok ```sql dalam response.")

    start_idx += len(start_token)
    end_idx = response.find(end_token, start_idx)
    if end_idx == -1:
        raise ValueError("Blok ```sql tidak ditutup dengan ```. Periksa output LLM.")
    
    sql_query = response[start_idx:end_idx].strip()
    if not sql_query:
        raise ValueError("SQL query kosong di dalam blok ```sql ... ```.")

    return sql_query

def execute_text2sql_response(conn, response: str):
    """
    Ekstrak SQL dari LLM output dan eksekusi query tersebut ke database.
    """
    try:
        sql_query = extract_sql_query_from_response(response)

        with conn.cursor() as cur:
            cur.execute(sql_query)
            rows = cur.fetchall()
            columns = [desc[0] for desc in cur.description]

        return rows, columns

    except Exception as e:
        raise RuntimeError(f"Error executing SQL: {e}")
