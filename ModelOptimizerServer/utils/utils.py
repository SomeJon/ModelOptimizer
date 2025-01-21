import pymysql

from dto.ExperimentDto import ExperimentDto
from dto.LayerDto import LayerDto
from dto.ModelDto import ModelDto
from dto.TestDto import TestDto
from dto.TestJobDto import TestJobDto
from utils.DB import DB


# Format results as an aligned table-like string
def format_as_aligned_table(headers, rows):
    # Calculate column widths
    column_widths = [len(header) for header in headers]
    for row in rows:
        column_widths = [max(width, len(str(value))) for width, value in zip(column_widths, row)]

    # Format the header
    header_line = " | ".join(f"{header:<{column_widths[i]}}" for i, header in enumerate(headers))
    separator_line = "-+-".join("-" * width for width in column_widths)

    # Format the rows
    row_lines = [
        " | ".join(f"{str(value):<{column_widths[i]}}" for i, value in enumerate(row))
        for row in rows
    ]

    # Combine everything into a single string
    return f"{header_line}\n{separator_line}\n" + "\n".join(row_lines)



