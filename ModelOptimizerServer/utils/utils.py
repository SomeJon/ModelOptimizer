from tabulate import tabulate
from sqlparse.tokens import DML


# Format results as a table-like string based on the specified format
def format_as_aligned_table(headers, rows, tablefmt='grid'):
    return tabulate(rows, headers, tablefmt=tablefmt)


def is_select_query(parsed):
    """
    Checks if the parsed SQL query is a single SELECT statement.

    Parameters:
    - parsed (list): Parsed SQL statements.

    Returns:
    - bool: True if it's a single SELECT statement, False otherwise.
    """
    if len(parsed) != 1:
        return False  # Only single statements are allowed

    stmt = parsed[0]
    if not stmt.tokens:
        return False  # Empty query

    # Find the first meaningful token
    first_token = stmt.token_first(skip_cm=True)
    if not first_token:
        return False

    # Check if the first token is a DML statement and is SELECT
    if first_token.ttype is DML and first_token.value.upper() == 'SELECT':
        return True

    return False