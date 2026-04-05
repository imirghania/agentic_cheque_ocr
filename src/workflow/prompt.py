from __future__ import annotations

ALLOWED_KEYS = {
    "bank_name",
    "bank_branch",
    "bank_info",
    "cheque_number",
    "date",
    "payee",
    "amount_in_words",
    "amount_in_numbers",
    "payer",
    "account_number",
    "micr_code",
    "sort_code",
}


def validate_keys(keys: list[str]) -> list[str]:
    invalid = set(keys) - ALLOWED_KEYS
    if invalid:
        raise ValueError(f"Invalid keys requested: {invalid}. Allowed keys: {ALLOWED_KEYS}")
    return keys


BANK_INFO_FIELDS = {
    "phone": {"type": "string", "description": "Bank phone number"},
    "fax": {"type": "string", "description": "Bank fax number"},
    "website": {"type": "string", "description": "Bank website URL"},
    "email": {"type": "string", "description": "Bank email address"},
    "post_box": {"type": "string", "description": "Bank postal box number"},
}


def _make_nullable(field_def: dict) -> dict:
    return {**field_def, "anyOf": [{"type": field_def["type"]}, {"type": "null"}]}


def build_resolve_keys_prompt(available_keys: list[str]) -> str:
    keys_list = "\n".join(f"  - {key}" for key in sorted(available_keys))
    return (
        f"You are an expert at understanding user requests for cheque data extraction.\n"
        f"Given a user's natural language prompt, determine which fields they want extracted.\n\n"
        f"Available fields:\n"
        f"{keys_list}\n\n"
        f"Return ONLY a JSON object with a 'requested_keys' array containing the matching field names.\n"
        f"If the user asks for all fields or doesn't specify, include all available fields.\n"
        f"Do not include any explanation or markdown."
    )


def build_dynamic_schema(requested_keys: list[str]) -> dict:
    field_map = {
        "bank_name": {"type": "string", "description": "Name of the bank"},
        "bank_branch": {"type": "string", "description": "Bank branch name or location"},
        "bank_info": {
            "type": "object",
            "description": "Bank contact details",
            "properties": {k: _make_nullable(v) for k, v in BANK_INFO_FIELDS.items()},
            "required": [],
            "additionalProperties": False,
        },
        "cheque_number": {"type": "string", "description": "Cheque number"},
        "date": {"type": "string", "description": "Date written on the cheque"},
        "payee": {"type": "string", "description": "Name of the payee (who the cheque is made out to)"},
        "amount_in_words": {"type": "string", "description": "Amount written out in words"},
        "amount_in_numbers": {"type": "string", "description": "Amount in numeric form"},
        "payer": {"type": "string", "description": "Name of the payer (account holder who issued the cheque)"},
        "account_number": {"type": "string", "description": "Account number of the payer"},
        "micr_code": {"type": "string", "description": "MICR code (Magnetic Ink Character Recognition) at the bottom of the cheque"},
        "sort_code": {"type": "string", "description": "Bank sort code"},
    }

    properties = {}
    for key in requested_keys:
        props = field_map[key].copy()
        if props["type"] != "object":
            props["anyOf"] = [{"type": "string"}, {"type": "null"}]
        properties[key] = props

    return {
        "type": "object",
        "title": "ChequeData",
        "properties": properties,
        "required": [],
        "additionalProperties": False,
    }


def build_extraction_prompt(ocr_text: str, requested_keys: list[str]) -> str:
    key_descriptions = {
        "bank_name": "Name of the bank",
        "bank_branch": "Bank branch name or location",
        "bank_info": "Bank contact details (phone, fax, website, email, post_box) as an object",
        "cheque_number": "Cheque number",
        "date": "Date written on the cheque",
        "payee": "Name of the payee (who the cheque is made out to)",
        "amount_in_words": "Amount written out in words",
        "amount_in_numbers": "Amount in numeric form",
        "payer": "Name of the payer (account holder who issued the cheque)",
        "account_number": "Account number of the payer",
        "micr_code": "MICR code (Magnetic Ink Character Recognition) at the bottom of the cheque",
        "sort_code": "Bank sort code",
    }

    keys_list = "\n".join(f"  - {key}: {key_descriptions[key]}" for key in requested_keys)

    return (
        f"You are an expert at extracting information from bank cheque images.\n"
        f"Below is the OCR-extracted text from a cheque image. The text preserves spatial layout.\n\n"
        f"--- OCR TEXT START ---\n"
        f"{ocr_text}\n"
        f"--- OCR TEXT END ---\n\n"
        f"Extract the following fields from the cheque text. If a field is not found, set it to null.\n\n"
        f"Fields to extract:\n"
        f"{keys_list}\n\n"
        f"Return ONLY a valid JSON object with the requested keys. Do not include any explanation or markdown."
    )
