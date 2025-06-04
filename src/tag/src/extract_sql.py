import re

def extract_sql(text: str) -> str:
    """Extract and format SQL query from text.

    Args:
        text (str): Raw text that may contain SQL code.

    Returns:
        str: Extracted SQL query.
    """
    # Extract SQL code enclosed in triple backticks
    pattern = r"(?:```sql|\\`\\`\\`)(.*?)(?:```|\\`\\`\\`)"
    matches = re.findall(pattern, text, re.DOTALL)
    sql_query = matches[0].strip() if matches else text

    # Check if there is a SELECT clause
    matches = re.search(r"SELECT", sql_query, re.IGNORECASE)
    sql_query = sql_query if matches else ""
    
    return sql_query