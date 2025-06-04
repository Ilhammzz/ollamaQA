def extract_sql_query_from_response(response: str) -> str:
    """
    Ekstrak SQLQuery dari response Text2SQL secara adaptif:
    - SQLQuery: kosong atau langsung isi
    - Dukungan blok ```sql
    - Dukungan multi-line SQL
    """
    lines = response.splitlines()
    sql_lines = []
    inside_sql_block = False
    found_sqlquery = False

    for line in lines:
        line = line.strip()

        if not found_sqlquery:
            if line.startswith("SQLQuery:"):
                found_sqlquery = True
                content_after = line[len("SQLQuery:"):].strip()

                # Kalau langsung ada query setelah SQLQuery:
                if content_after:
                    # Kalau langsung ```sql
                    if content_after.startswith("```sql"):
                        inside_sql_block = True
                        remainder = content_after[len("```sql"):].strip()
                        if remainder:
                            sql_lines.append(remainder)
                    elif content_after.startswith("```"):
                        inside_sql_block = True
                    else:
                        # Langsung query singkat
                        sql_lines.append(content_after)
                continue
        elif found_sqlquery and not inside_sql_block:
            # Sudah nemu SQLQuery:, belum mulai capture, cari ```sql
            if line.startswith("```sql") or line.startswith("```"):
                inside_sql_block = True
                continue
            elif line:  # Kalau langsung SELECT tanpa ```sql
                sql_lines.append(line)
        elif inside_sql_block:
            if line.startswith("```"):
                # Tutup blok SQL
                inside_sql_block = False
                break
            sql_lines.append(line)

    if not sql_lines:
        raise ValueError(f"Tidak menemukan SQLQuery yang valid dalam response:\n{response}")

    sql_query = "\n".join(sql_lines).strip()
    return sql_query

def execute_text2sql_response(conn, response: str):
    """
    Dari hasil Text2SQL, extract SQLQuery, bersihkan, eksekusi ke database, dan kembalikan hasil.

    Args:
        conn: koneksi database PostgreSQL
        response: response Text2SQL dari generate_sql()

    Returns:
        rows (list of tuple), columns (list of str)
    """
    try:
        # 1. Extract dan bersihkan SQLQuery
        sql_query = extract_sql_query_from_response(response)

        # 2. Eksekusi ke database
        cur = conn.cursor()
        cur.execute(sql_query)
        rows = cur.fetchall()
        columns = [desc[0] for desc in cur.description]
        cur.close()

        return rows, columns

    except Exception as e:
        raise RuntimeError(f"Error executing SQL: {e}")
