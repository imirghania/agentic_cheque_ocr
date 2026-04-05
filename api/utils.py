from api.schemas import ExtractResponse


def format_markdown(result: ExtractResponse) -> str:
    lines = ["# Cheque Extraction Results\n"]
    data = result.extracted_data
    rows = []
    for key, value in data.items():
        label = key.replace("_", " ").title()
        if isinstance(value, dict):
            lines.append(f"### {label}\n")
            for k, v in value.items():
                rows.append((k.replace("_", " ").title(), v if v is not None else "N/A"))
        else:
            rows.append((label, value if value is not None else "N/A"))
    if rows:
        lines.append("| Field | Value |")
        lines.append("|-------|-------|")
        for label, value in rows:
            lines.append(f"| {label} | {value} |")
        lines.append("")
    if result.ocr_confidence is not None:
        lines.append(f"**OCR Confidence:** {result.ocr_confidence:.1f}%\n")
    if result.error:
        lines.append(f"**Error:** {result.error}\n")
    return "\n".join(lines)