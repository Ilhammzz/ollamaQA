def load_schema(conn):
    """
    Mengambil skema database (nama tabel dan kolom).
    """
    cur = conn.cursor()
    cur.execute("""
        SELECT table_name, column_name, data_type
        FROM information_schema.columns
        WHERE table_schema = 'public'
        ORDER BY table_name, ordinal_position;
    """)
    rows = cur.fetchall()
    cur.close()

    # schema = ""
    # current_table = None
    # for table, column in rows:
    #     if table != current_table:
    #         schema += f"\nTable {table}:\n"
    #         current_table = table
    #     schema += f"  - {column}\n"
    
    # return schema.strip()

    schema = ""
    current_table = None
    for table, column, dtype in rows:
        if table != current_table:
            schema += f"\nTable {table}:\n"
            current_table = table
        schema += f"  - {column} ({dtype})\n"
    
    return schema.strip()
