"""Build small, valid text PDFs in memory so tests need no sample files."""


def _escape(text: str) -> str:
    """Escape the characters that are special inside a PDF string."""
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def make_pdf(pages: list[str]) -> bytes:
    """Create a PDF with one line of text per page.

    Args:
        pages: The text of each page (ASCII only). An empty string makes a
            page without any text.

    Returns:
        The bytes of a valid PDF file.
    """
    page_count = len(pages)
    first_page_id = 4
    bodies: dict[int, str] = {
        1: "<< /Type /Catalog /Pages 2 0 R >>",
        3: "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    }
    kids = " ".join(f"{first_page_id + 2 * index} 0 R" for index in range(page_count))
    bodies[2] = f"<< /Type /Pages /Kids [{kids}] /Count {page_count} >>"
    for index, text in enumerate(pages):
        page_id = first_page_id + 2 * index
        stream = f"BT /F1 12 Tf 72 720 Td ({_escape(text)}) Tj ET"
        bodies[page_id] = (
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 3 0 R >> >> /Contents {page_id + 1} 0 R >>"
        )
        bodies[page_id + 1] = f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream"

    output = b"%PDF-1.4\n"
    offsets: dict[int, int] = {}
    for object_id in sorted(bodies):
        offsets[object_id] = len(output)
        output += f"{object_id} 0 obj\n{bodies[object_id]}\nendobj\n".encode("latin-1")
    xref_offset = len(output)
    object_count = max(bodies) + 1
    output += f"xref\n0 {object_count}\n0000000000 65535 f \n".encode("latin-1")
    for object_id in range(1, object_count):
        output += f"{offsets[object_id]:010d} 00000 n \n".encode("latin-1")
    output += (
        f"trailer\n<< /Size {object_count} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n"
    ).encode("latin-1")
    return output
