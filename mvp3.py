"""
MVP 3: Batch Record Extraction with Specificity

This version improves upon the previous MVP by adding
a table which shows all the files which have been added using upserts
a CSV which acts as an external source of memory
an option to very accurately obtain data from a specific file structure
tables which display this specific data, not using upserts

Goal:
Enhance previous MVP by adding
different ways of obtaining and displaying information
"""
import json
import os
import re
from collections import Counter
import hashlib

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from mvp3_extractor import AzureDocIntExtractor

WORD_CONFIDENCE_THRESHOLD = 0.75
FIELD_CONFIDENCE_THRESHOLD = 0.75
MAX_LOW_CONFIDENCE_WORDS_TO_SHOW = 15

load_dotenv()

if "processed" not in st.session_state:
    st.session_state.processed = False

if "raw_result" not in st.session_state:
    st.session_state.raw_result = None

if "normalized_result" not in st.session_state:
    st.session_state.normalized_result = None

if "parsed_tables" not in st.session_state:
    st.session_state.parsed_tables = []

if "summary_df" not in st.session_state:
    st.session_state.summary_df = None

if "document_name" not in st.session_state:
    st.session_state.document_name = None

if "document_hash" not in st.session_state:
    st.session_state.document_hash = None

if "product_details" not in st.session_state:
    st.session_state.product_details = {
        'tablet_number': [],
        'batch_number': [],
        'manufacturing_date': [],
        'expiry_date': []
    }

if "document_details" not in st.session_state:
    st.session_state.document_details = {
        'prepare_sign': [],
        'prepare_sign_date': [],
        'approve_sign': [],
        'approve_sign_date': []
    }

if "preparer_details" not in st.session_state:
    st.session_state.preparer_details = {
        'name': []
    }

if "approver_details" not in st.session_state:
    st.session_state.approver_details = {
        'name': []
    }


st.set_page_config(page_title="Azure Batch Record Extraction MVP", layout="wide")

st.title("Biopharma Batch Record Extraction - Azure Version")
st.write(
    "Upload a PDF batch record and process it with Azure Document Intelligence. "
    "This version supports prebuilt OCR/layout analysis."
)


@st.cache_resource
def load_azure_extractor():
    return AzureDocIntExtractor()


def extract_with_patterns(text: str, patterns: list[str]):
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return None


def polygon_to_bbox(polygon: list) -> dict | None:
    """
    Convert Azure polygon [x1,y1,x2,y2,x3,y3,x4,y4] to bounding box.
    """
    if not polygon or len(polygon) < 8:
        return None

    xs = polygon[0::2]
    ys = polygon[1::2]

    return {
        "x_min": min(xs),
        "y_min": min(ys),
        "x_max": max(xs),
        "y_max": max(ys),
    }


def word_center_in_bbox(word_polygon, cell_bbox) -> bool:
    """
    Check whether the center of a word polygon lies inside a cell bbox.
    """
    if not word_polygon or not cell_bbox:
        return False

    xs = word_polygon[0::2]
    ys = word_polygon[1::2]

    x_center = sum(xs) / len(xs)
    y_center = sum(ys) / len(ys)

    return (
        cell_bbox["x_min"] <= x_center <= cell_bbox["x_max"]
        and cell_bbox["y_min"] <= y_center <= cell_bbox["y_max"]
    )


def recover_cell_text_from_words(cell: dict, pages: list[dict]) -> str:
    """
    Recover cell text by finding page words whose centers fall inside the cell polygon.
    """
    bounding_regions = cell.get("bounding_regions", [])
    if not bounding_regions:
        return ""

    recovered_words = []

    for region in bounding_regions:
        pagenum = region.get("page_number")
        polygon = region.get("polygon")
        cell_bbox = polygon_to_bbox(polygon)

        if not cell_bbox:
            continue

        matching_page = next(
            (p for p in pages if p.get("page_number") == pagenum),
            None
        )

        if not matching_page:
            continue

        for word in matching_page.get("words", []):
            word_polygon = word.get("polygon")
            if word_center_in_bbox(word_polygon, cell_bbox):
                recovered_words.append(word.get("content", ""))

    return " ".join(w for w in recovered_words if w).strip()


def parse_azure_table(t: dict, pages: list[dict]) -> list[dict]:
    """
    Convert one Azure table object into a list of row dictionaries.

    Uses cell.content first.
    If cell.content is empty, tries to recover text from page words
    inside the cell bounding box.
    Assumes row 0 contains headers.
    """
    rows = {}

    for cell in t.get("cells", []):
        r = cell["row_index"]
        c = cell["column_index"]

        text = (cell.get("content") or "").strip()

        if not text:
            text = recover_cell_text_from_words(cell, pages)

        if r not in rows:
            rows[r] = {}

        column_name = c
        rows[r][column_name] = text

    pre_table = list(rows.values())

    return pre_table


def parse_all_azure_tables(rr: dict) -> list[dict]:
    """
    Parse all Azure tables into structured row-based output.
    """
    parsed_t = []
    pages = rr.get("pages", [])

    for idx, t in enumerate(rr.get("tables", []), start=1):
        parsed_rows = parse_azure_table(t, pages)

        parsed_t.append({
            "table_index": idx,
            "row_count": t.get("row_count"),
            "column_count": t.get("column_count"),
            "rows": parsed_rows,
        })

    return parsed_t


def classify_page(text: str) -> str:
    if "batch number" in text.lower():
        return "header_page"

    if "material" in text.lower():
        return "material_page"

    if "time" in text.lower() and "temperature" in text.lower():
        return "process_log_page"

    return "unknown_page"


def extract_handwritten_spans(rr: dict) -> list[dict]:
    handwritten = []

    full_content = rr.get("content", "")
    styles = rr.get("styles", [])

    for style in styles:
        if style.get("is_handwritten") is True:
            for span in style.get("spans", []):
                offset = span.get("offset", 0)
                length = span.get("length", 0)
                text = full_content[offset:offset + length]

                handwritten.append({
                    "text": text,
                    "offset": offset,
                    "length": length,
                    "confidence": style.get("confidence"),
                })

    return handwritten


def get_low_confidence_words(p: dict, threshold: float = WORD_CONFIDENCE_THRESHOLD) -> list[dict]:
    """
    Return words on a page whose confidence is below the threshold.
    """
    lowconf_words = []

    for word in p.get("words", []):
        confidence = word.get("confidence")
        if confidence is not None and confidence < threshold:
            lowconf_words.append({
                "content": word.get("content", ""),
                "confidence": confidence,
                "polygon": word.get("polygon"),
            })

    return lowconf_words


def estimate_field_confidence(field_value: str | None, p: dict) -> float | None:
    """
    Estimate field confidence by matching the extracted value against page words.
    Returns the best matching word confidence if found.
    """
    if not field_value:
        return None

    field_value_clean = str(field_value).strip().lower()

    best_confidence = None

    for word in p.get("words", []):
        word_text = str(word.get("content", "")).strip().lower()
        confidence = word.get("confidence")

        if not word_text or confidence is None:
            continue

        if word_text == field_value_clean or field_value_clean in word_text or word_text in field_value_clean:
            if best_confidence is None or confidence > best_confidence:
                best_confidence = confidence

    return best_confidence


def build_field_review_flags(fields: dict, p: dict, threshold: float = FIELD_CONFIDENCE_THRESHOLD) -> list[dict]:
    """
    Build review flags for extracted fields with low estimated confidence.
    """
    flags = []

    for field_name, field_value in fields.items():
        if field_value in [None, ""]:
            continue

        estimated_confidence = estimate_field_confidence(field_value, p)

        if estimated_confidence is not None and estimated_confidence < threshold:
            flags.append({
                "field_name": field_name,
                "field_value": field_value,
                "confidence": estimated_confidence,
                "reason": "Low OCR confidence for extracted field",
            })

    return flags


def normalize_simpletest(result: dict) -> dict:
    normalized = {
        "pages": [],
        "validation_warnings": [],
    }

    pages = result.get("pages", [])
    if len(pages) != 1:
        raise ValueError('"SimpleTest" is not the correct document setting.\n'
                         'Please choose another setting.')

    p = pages[0]
    # page_layout = [_ for _ in p.get("lines", [])]
    page_lines = [line["content"] for line in p.get("lines", [])]
    page_text = "\n".join(page_lines)

    tables = result.get("tables", [])

    if len(tables) != 2:
        raise ValueError('"SimpleTest" is not the correct document setting.\n'
                         'Please choose another setting.')

    product_details = {
        'tablet_number': None,
        'batch_number': None,
        'manufacturing_date': None,
        'expiry_date': None
    }

    document_details = {
        'prepare_sign': None,
        'prepare_sign_date': None,
        'approve_sign': None,
        'approve_sign_date': None
    }

    preparer_details = {
        'name': None
    }

    approver_details = {
        'name': None
    }

    first_table, second_table = tables[0], tables[1]

    correct_tables = (first_table['row_count'] == 3 and
                      first_table['column_count'] == 4 and
                      second_table['row_count'] == 2 and
                      second_table['column_count'] == 3)

    if not correct_tables:
        raise ValueError('"SimpleTest" is not the correct document setting.\n'
                         'Please choose another setting.')

    for cell in first_table['cells']:
        match (cell['row_index'], cell['column_index']):
            case (1, 1):
                content = re.search(r'(\w+) Production Manager', cell['content'])
                if content:
                    success = content.group(1).strip()
                    preparer_details['name'] = success
                else:
                    normalized['validation_warnings'].append(
                        "May be missing production manager name."
                    )
            case (1, 2):
                content = cell['content']
                if content:
                    document_details['prepare_sign'] = content
                else:
                    normalized['validation_warnings'].append(
                        "May be missing production manager signature."
                    )
            case (1, 3):
                content = cell['content']
                if content:
                    document_details['prepare_sign_date'] = content
                else:
                    normalized['validation_warnings'].append(
                        "May be missing date of production manager signature."
                    )
            case (2, 1):
                content = re.search(r'(\w+) QA Manager', cell['content'])
                if content:
                    success = content.group(1).strip()
                    approver_details['name'] = success
                else:
                    normalized['validation_warnings'].append(
                        "May be missing QA manager name."
                    )
            case (2, 2):
                content = cell['content']
                if content:
                    document_details['approve_sign'] = content
                else:
                    normalized['validation_warnings'].append(
                        "May be missing QA manager signature."
                    )
            case (2, 3):
                content = cell['content']
                if content:
                    document_details['approve_sign_date'] = content
                else:
                    normalized['validation_warnings'].append(
                        "May be missing date of production manager signature."
                    )

    tablet_there = re.search(r'\nTablets No:\n(\d+)\n', page_text)

    if tablet_there:
        tablet_number = tablet_there.group(1).strip()
        product_details['tablet_number'] = tablet_number
    else:
        normalized['validation_warnings'].append(
            "May be missing tablet number."
        )

    for cell in second_table['cells']:
        match (cell['row_index'], cell['column_index']):
            case (1, 0):
                content = cell['content']
                if content:
                    product_details['batch_number'] = content
                else:
                    normalized['validation_warnings'].append(
                        "May be missing batch number."
                    )
            case(1, 1):
                content = cell['content']
                if content:
                    product_details['manufacturing_date'] = content
                else:
                    normalized['validation_warnings'].append(
                        "May be missing manufacturing date."
                    )
            case (1, 2):
                content = cell['content']
                if content:
                    product_details['expiry_date'] = content
                else:
                    normalized['validation_warnings'].append(
                        "May be missing expiry date."
                    )

    normalized["product_details"] = product_details
    normalized["document_details"] = document_details
    normalized["preparer_details"] = preparer_details
    normalized["approver_details"] = approver_details

    pagenum = p['page_number']
    page_type = 'header page'

    fields = {}

    low_confidence_words = get_low_confidence_words(p)
    field_review_flags = build_field_review_flags(fields, p)

    if low_confidence_words:
        normalized['validation_warnings'].append(
            f"{len(low_confidence_words)} low-confidence OCR word(s) detected"
        )

    if field_review_flags:
        normalized['validation_warnings'].append(
            f"{len(field_review_flags)} extracted field(s) flagged for review"
        )

    normalized["pages"].append(
        {
            'page_number': pagenum,
            'page_type': page_type,
            "ocr_text": page_text,
            "fields": fields,
            "review_flags": {
                "low_confidence_words": low_confidence_words[:MAX_LOW_CONFIDENCE_WORDS_TO_SHOW],
                "field_flags": field_review_flags,
            },
        }
    )

    return normalized


def normalize_prebuilt_result(result: dict) -> dict:
    """
    Normalize Azure prebuilt-read or prebuilt-layout output into a simple schema.
    """
    normalized = {
        "document_batch_number": None,
        "document_lot_number": None,
        "pages": [],
        "validation_warnings": [],
    }

    detected_batch_numbers = []
    detected_lot_numbers = []

    for p in result.get("pages", []):
        pagenum = p["page_number"]
        # page_layout = [_ for _ in p.get("lines", [])]
        page_lines = [line["content"] for line in p.get("lines", [])]
        page_text = "\n".join(page_lines)

        page_type = classify_page(page_text)

        batch_number = extract_with_patterns(
            page_text,
            [
                r"Batch\s*Number[:\-]?\s*([A-Za-z0-9\-\/]+)",
                r"Batch\s*Record\s*:\n*\s*([A-Za-z0-9\-\/]+)",
                r"Batch\s*No[:\-]?\s*([A-Za-z0-9\-\/]+)",
                r"BMR\s*No\s*.:\s*([A-Za-z0-9\-\/]+)"
            ],
        )

        lot_number = extract_with_patterns(
            page_text,
            [
                r"Lot\s*Number[:\-]?\s*([A-Za-z0-9\-\/]+)",
                r"Lot\s*No[:\-]?\s*([A-Za-z0-9\-\/]+)",
                r"Lot[:\-]?\s*([A-Za-z0-9\-\/]+)",
            ],
        )

        date_value = extract_with_patterns(
            page_text,
            [
                r"Date[:\-]?\s*([0-9]{1,2}[\/\-][0-9]{1,2}[\/\-][0-9]{2,4})",
            ],
        )

        operator = extract_with_patterns(
            page_text,
            [
                r"Operator[:\-]?\s*([A-Za-z]{1,20})",
                r"Operator\s*Initials[:\-]?\s*([A-Za-z]{1,10})",
                r"Initials[:\-]?\s*([A-Za-z]{1,10})",
            ],
        )

        if batch_number:
            detected_batch_numbers.append(batch_number)

        if lot_number:
            detected_lot_numbers.append(lot_number)

        fields = {
            "batch_number": batch_number,
            "lot_number": lot_number,
            "date": date_value,
            "operator": operator,
        }

        low_confidence_words = get_low_confidence_words(p)
        field_review_flags = build_field_review_flags(fields, p)

        page_warnings = []

        if low_confidence_words:
            page_warnings.append(
                f"{len(low_confidence_words)} low-confidence OCR word(s) detected on page {pagenum}"
            )

        if field_review_flags:
            page_warnings.append(
                f"{len(field_review_flags)} extracted field(s) flagged for review on page {pagenum}"
            )

        normalized["pages"].append(
            {
                "page_number": pagenum,
                "page_type": page_type,
                "ocr_text": page_text,
                "fields": fields,
                "review_flags": {
                    "low_confidence_words": low_confidence_words[:MAX_LOW_CONFIDENCE_WORDS_TO_SHOW],
                    "field_flags": field_review_flags,
                },
                "page_warnings": page_warnings,
            }
        )

    if detected_batch_numbers:
        batch_counts = Counter(detected_batch_numbers)
        normalized["document_batch_number"] = batch_counts.most_common(1)[0][0]
        if len(batch_counts) > 1:
            normalized["validation_warnings"].append(
                f"Multiple batch numbers detected: {dict(batch_counts)}"
            )

    if detected_lot_numbers:
        lot_counts = Counter(detected_lot_numbers)
        normalized["document_lot_number"] = lot_counts.most_common(1)[0][0]
        if len(lot_counts) > 1:
            normalized["validation_warnings"].append(
                f"Multiple lot numbers detected: {dict(lot_counts)}"
            )

    for p in normalized["pages"]:
        if not p["fields"]["batch_number"]:
            p["fields"]["batch_number"] = normalized["document_batch_number"]
        if not p["fields"]["lot_number"]:
            p["fields"]["lot_number"] = normalized["document_lot_number"]
        for warn in p.get("page_warnings", []):
            normalized["validation_warnings"].append(warn)

    return normalized


def normalize_prebuilt_with_document(rr: dict, doc_choice: str) -> dict:
    """
    Redirect Azure prebuilt model normalization to the proper channel.
    """
    if doc_choice == "SimpleTest":
        return normalize_simpletest(rr)
    else:
        return normalize_prebuilt_result(rr)


def normalize_custom_result(result: dict) -> dict:
    """
    Normalize Azure custom model output into the same simple schema.
    """
    normalized = {
        "document_batch_number": None,
        "document_lot_number": None,
        "pages": [],
        "validation_warnings": [],
    }

    for idx, doc in enumerate(result.get("documents", []), start=1):
        fields = doc.get("fields", {})

        def get_field_value(*names):
            for name in names:
                if name in fields:
                    f_name = fields[name]
                    for k in ["value_string", "value_date", "value_time", "content", "value_number"]:
                        v = f_name.get(k)
                        if v is not None:
                            return str(v)
            return None

        batch_number = get_field_value("batch_number", "BatchNumber", "Batch_Number")
        lot_number = get_field_value("lot_number", "LotNumber", "Lot_Number")
        date_value = get_field_value("date", "Date", "ManufactureDate")
        operator = get_field_value("operator", "Operator", "OperatorInitials")

        material_name = get_field_value("material_name", "MaterialName")
        material_lot = get_field_value("material_lot", "MaterialLot")
        quantity = get_field_value("quantity", "Quantity", "QuantityAdded")

        if batch_number and not normalized["document_batch_number"]:
            normalized["document_batch_number"] = batch_number

        if lot_number and not normalized["document_lot_number"]:
            normalized["document_lot_number"] = lot_number

        all_field_names = ["batch_number",
                           "lot_number",
                           "date",
                           "operator",
                           "material_name",
                           "material_lot",
                           "quantity"]

        all_field_flags = []

        for field_name in all_field_names:
            candidate_names = [field_name,
                               field_name.title().replace("_", ""),
                               field_name.replace("_", "")]

            for candidate_name in candidate_names:
                if candidate_name in fields:
                    field_obj = fields[candidate_name]
                    confidence = field_obj.get("confidence")
                    value = None

                    for key in ["value_string", "value_date", "value_time", "content", "value_number"]:
                        if field_obj.get(key) is not None:
                            value = str(field_obj.get(key))
                            break

                    if value is not None and confidence is not None and confidence < FIELD_CONFIDENCE_THRESHOLD:
                        all_field_flags.append({
                            "field_name": field_name,
                            "field_value": value,
                            "confidence": confidence,
                            "reason": "Low custom model confidence",
                        })
                    break

        normalized["pages"].append(
            {
                "page_number": idx,
                "page_type": doc.get("doc_type", "custom_page"),
                "ocr_text": None,
                "fields": {
                    "batch_number": batch_number,
                    "lot_number": lot_number,
                    "date": date_value,
                    "operator": operator,
                    "material_name": material_name,
                    "material_lot": material_lot,
                    "quantity": quantity,
                },
                "confidence": doc.get("confidence"),
                "review_flags": {
                    "field_flags": all_field_flags,
                },
                "page_warnings": [
                    f"{len(all_field_flags)} extracted field(s) flagged for review on page {idx}"
                ] if all_field_flags else [],
            }
        )

        for p in normalized["pages"]:
            if not p["fields"]["batch_number"]:
                p["fields"]["batch_number"] = normalized["document_batch_number"]
            if not p["fields"]["lot_number"]:
                p["fields"]["lot_number"] = normalized["document_lot_number"]
            for warn in p.get("page_warnings", []):
                normalized["validation_warnings"].append(warn)

    return normalized


def normalize_custom_with_document(rr: dict, doc_choice: str) -> dict:
    """
    Redirect Azure custom model normalization to the proper channel.
    """
    return normalize_custom_result(rr)  # may change later


def build_summary_dataframe(normalized: dict) -> pd.DataFrame:
    rows = []
    for p in normalized.get("pages", []):
        row = {
            "page_number": p.get("page_number"),
            "page_type": p.get("page_type"),
            "needs_review": bool(
                p.get("review_flags", {}).get("field_flags")
                or p.get("review_flags", {}).get("low_confidence_words")
            ),
        }
        row.update(p.get("fields", {}))
        rows.append(row)
    return pd.DataFrame(rows)


def generate_postgresql_from_dataframes(dataframes: dict[str, pd.DataFrame]) -> str:
    sql_parts = []

    for table_name, df in dataframes.items():
        # --- CREATE TABLE ---
        col_definitions = []
        for col in df.columns:
            dtype = df[col].dtype
            if pd.api.types.is_integer_dtype(dtype):
                sql_type = "INTEGER"
            elif pd.api.types.is_float_dtype(dtype):
                sql_type = "NUMERIC"
            elif pd.api.types.is_bool_dtype(dtype):
                sql_type = "BOOLEAN"
            elif pd.api.types.is_datetime64_any_dtype(dtype):
                sql_type = "TIMESTAMP"
            else:
                sql_type = "TEXT"

            col_definitions.append(f'    "{col}" {sql_type}')

        create_stmt = (
            f'CREATE TABLE IF NOT EXISTS "{table_name}" (\n'
            + ",\n".join(col_definitions)
            + "\n);"
        )
        sql_parts.append(create_stmt)

        # --- INSERT ROWS ---
        if not df.empty:
            col_names = ", ".join(f'"{col}"' for col in df.columns)

            row_values = []
            for _, row in df.iterrows():
                formatted = []
                for col in df.columns:
                    val = row[col]
                    if val is None or (isinstance(val, float) and pd.isna(val)):
                        formatted.append("NULL")
                    elif isinstance(val, bool):
                        formatted.append("TRUE" if val else "FALSE")
                    elif isinstance(val, (int, float)):
                        formatted.append(str(val))
                    else:
                        escaped = str(val).replace("'", "''")
                        formatted.append(f"'{escaped}'")

                row_values.append("    (" + ", ".join(formatted) + ")")

            insert_stmt = (
                f'INSERT INTO "{table_name}" ({col_names}) VALUES\n'
                + ",\n".join(row_values)
                + ";"
            )
            sql_parts.append(insert_stmt)

    return "\n\n".join(sql_parts)


azure_extractor = load_azure_extractor()

with st.sidebar:
    st.header("Azure Settings")
    endpoint_present = bool(os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT"))
    key_present = bool(os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY"))
    st.write(f"Endpoint loaded: {'Yes' if endpoint_present else 'No'}")
    st.write(f"Key loaded: {'Yes' if key_present else 'No'}")

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

pdf_bytes = None
current_doc_hash = None

if uploaded_file is not None:
    pdf_bytes = uploaded_file.read()
    current_doc_hash = hashlib.md5(pdf_bytes).hexdigest()

    # checks if a different document has been uploaded
    if (
        st.session_state.document_hash is not None
        and st.session_state.document_hash != current_doc_hash
    ):
        st.session_state.processed = False
        st.session_state.raw_result = None
        st.session_state.normalized_result = None
        st.session_state.parsed_tables = []
        st.session_state.summary_df = None

    st.session_state.document_hash = current_doc_hash
    st.session_state.document_name = uploaded_file.name

with st.form("process_form"):
    model_choice = st.selectbox(
        "Choose Azure model",
        ["prebuilt-layout", "prebuilt-read"],
    )

    document_choice = st.selectbox(
        "Choose specified document type",
        ["N/A", "SimpleTest"],
    )

    pages_to_process = st.text_input(
        "Pages to process (optional)",
        placeholder="Example: 1-3 or 1,3,5",
    )

    custom_model_id = ""
    if model_choice == "custom-model":
        custom_model_id = st.text_input(
            "Custom model ID",
            value=os.getenv("AZURE_DOCUMENT_INTELLIGENCE_CUSTOM_MODEL_ID", ""),
        )

    process_clicked = st.form_submit_button("Process Document")

if uploaded_file is not None:
    if process_clicked and pdf_bytes is not None:
        with st.spinner("Sending document to Azure Document Intelligence..."):
            try:
                page_count = azure_extractor.get_pdf_page_count(pdf_bytes)

                if model_choice == "prebuilt-read":
                    if page_count > 50 and not pages_to_process:
                        raw_result = azure_extractor.analyze_read_chunked(pdf_bytes, chunk_size=50)
                    else:
                        raw_result = azure_extractor.analyze_read(
                            pdf_bytes,
                            pages=pages_to_process or None,
                        )
                    normalized_result = normalize_prebuilt_with_document(raw_result, document_choice)

                elif model_choice == "prebuilt-layout":
                    if page_count > 50 and not pages_to_process:
                        raw_result = azure_extractor.analyze_layout_chunked(pdf_bytes, chunk_size=50)
                    else:
                        raw_result = azure_extractor.analyze_layout(
                            pdf_bytes,
                            pages=pages_to_process or None,
                        )
                    normalized_result = normalize_prebuilt_with_document(raw_result, document_choice)

                else:
                    if page_count > 50 and not pages_to_process:
                        raw_result = azure_extractor.analyze_custom_chunked(
                            pdf_bytes,
                            model_id=custom_model_id,
                            chunk_size=50,
                        )
                    else:
                        raw_result = azure_extractor.analyze_custom(
                            pdf_bytes,
                            model_id=custom_model_id,
                            pages=pages_to_process or None,
                        )
                    normalized_result = normalize_custom_result(raw_result)

                parsed_tables = []
                if model_choice == "prebuilt-layout":
                    parsed_tables = parse_all_azure_tables(raw_result)

                st.success("Document processed successfully.")

                df = build_summary_dataframe(normalized_result)

                output_package = {
                    "normalized_result": normalized_result,
                    "parsed_tables": parsed_tables,
                }

                normalized_json = json.dumps(output_package, indent=2)
                raw_json = json.dumps(raw_result, indent=2)

                if 'product_details' in normalized_result:
                    for x in normalized_result['product_details']:
                        st.session_state.product_details[x].append(normalized_result['product_details'][x])
                    for x in normalized_result['document_details']:
                        st.session_state.document_details[x].append(normalized_result['document_details'][x])
                    for x in normalized_result['preparer_details']:
                        st.session_state.preparer_details[x].append(normalized_result['preparer_details'][x])
                    for x in normalized_result['approver_details']:
                        st.session_state.approver_details[x].append(normalized_result['approver_details'][x])

                st.session_state.raw_result = raw_result
                st.session_state.normalized_result = normalized_result
                st.session_state.parsed_tables = parsed_tables
                st.session_state.summary_df = df
                st.session_state.document_name = uploaded_file.name
                st.session_state.processed = True

                # Record this file in mvp3_dataframe.csv (avoid duplicate entries
                # if the same document is re-processed within the same session).
                log_entry = {
                    "file_name": uploaded_file.name,
                    "batch_number": normalized_result.get("document_batch_number"),
                    "lot_number": normalized_result.get("document_lot_number"),
                }

                log_csv_path = "mvp3_dataframe.csv"
                if os.path.exists(log_csv_path):
                    log_df = pd.read_csv(log_csv_path)
                    if uploaded_file.name in log_df["file_name"].values:
                        log_df.loc[log_df["file_name"] == uploaded_file.name, ["batch_number", "lot_number"]] = (
                            log_entry["batch_number"],
                            log_entry["lot_number"],
                        )
                    else:
                        log_df = pd.concat(
                            [log_df, pd.DataFrame([log_entry])],
                            ignore_index=True,
                        )
                else:
                    log_df = pd.DataFrame([log_entry], columns=["file_name", "batch_number", "lot_number"])
                log_df.to_csv(log_csv_path, index=False)

            except Exception as e:
                st.error(f"Error processing document: {e}")

if st.session_state.processed and st.session_state.normalized_result is not None:
    raw_result = st.session_state.raw_result
    normalized_result = st.session_state.normalized_result
    parsed_tables = st.session_state.parsed_tables
    df = st.session_state.summary_df
    document_name = st.session_state.document_name

    st.success(f"Showing saved results for: {document_name}")

    col1, col2 = st.columns(2)

    with col1:
        with st.expander("Show Normalized Batch Record Output"):
            st.json(normalized_result)

    with col2:
        with st.expander("Show Raw Azure Output"):
            st.json(raw_result)

    if normalized_result.get("validation_warnings"):
        with st.expander("Show Validation Warnings"):
            for warning in normalized_result["validation_warnings"]:
                st.warning(warning)

    flag_count = 0

    with st.expander("Show Low-Confidence OCR Words"):
        for page in normalized_result.get("pages", []):
            page_number = page.get("page_number")
            review_flags = page.get("review_flags", {})
            field_flags = review_flags.get("field_flags", [])
            low_conf_words = review_flags.get("low_confidence_words", [])

            if field_flags or low_conf_words:
                flag_count += 1
                with st.expander(f"Page {page_number} review flags"):
                    if field_flags:
                        st.write("### Field Flags")
                        st.json(field_flags)

                    if low_conf_words:
                        st.write("### Low-Confidence OCR Words")
                        st.json(low_conf_words)

    if flag_count == 0:
        st.success("No confidence-based review flags found.")

    if document_choice == 'SimpleTest':
        with st.expander("Show SQL Tables"):
            st.write("Product Details:")
            details_df1 = pd.DataFrame(st.session_state.product_details)
            edited_df1 = st.data_editor(details_df1, width='stretch')
            st.session_state.product_details = edited_df1.to_dict(orient="list")
            st.write("Document Details:")
            details_df2 = pd.DataFrame(st.session_state.document_details)
            edited_df2 = st.data_editor(details_df2, width='stretch')
            st.session_state.document_details = edited_df2.to_dict(orient="list")
            st.write("Preparer Details:")
            details_df3 = pd.DataFrame(st.session_state.preparer_details)
            edited_df3 = st.data_editor(details_df3, width='stretch')
            st.session_state.preparer_details = edited_df3.to_dict(orient="list")
            st.write("Approver Details:")
            details_df4 = pd.DataFrame(st.session_state.approver_details)
            edited_df4 = st.data_editor(details_df4, width='stretch')
            st.session_state.approver_details = edited_df4.to_dict(orient="list")
            all_dataframes = {
                'product_details': edited_df1,
                'document_details': edited_df2,
                'preparer_details': edited_df3,
                'approver_details': edited_df4
            }
            sql_creation_file = generate_postgresql_from_dataframes(all_dataframes)

            st.download_button(
                label="Download PostgreSQL",
                data=sql_creation_file,
                file_name=f"{document_name}_postgre.sql",
                mime="text/plain",
                use_container_width=True,
            )

    elif parsed_tables:
        with st.expander("Show Parsed Azure Tables"):

            for table in parsed_tables:
                if 'table_index' in table:
                    st.write(
                        f"Raw Table {table['table_index']} "
                        f"({table['row_count']} rows x {table['column_count']} columns)"
                    )

                    table_rows = table["rows"]
                    if table_rows:
                        table_df = pd.DataFrame(table_rows)
                        st.dataframe(table_df, width='stretch')
                    else:
                        st.info("No parsed rows found for this table.")

if st.session_state.processed and st.session_state.normalized_result is not None:
    raw_result = st.session_state.raw_result
    normalized_result = st.session_state.normalized_result
    parsed_tables = st.session_state.parsed_tables
    df = st.session_state.summary_df
    document_name = st.session_state.document_name

    output_package = {
        "normalized_result": normalized_result,
        "parsed_tables": parsed_tables,
    }

    normalized_json = json.dumps(output_package, indent=2)
    raw_json = json.dumps(raw_result, indent=2)
    csv_output = df.to_csv(index=False)

    st.download_button(
        label="Download Normalized JSON",
        data=normalized_json,
        file_name=f"{document_name}_normalized.json",
        mime="application/json",
    )

    st.download_button(
        label="Download Raw Azure JSON",
        data=raw_json,
        file_name=f"{document_name}_raw_azure.json",
        mime="application/json",
    )

    st.download_button(
        label="Download CSV Summary",
        data=csv_output,
        file_name=f"{document_name}_summary.csv",
        mime="text/csv",
    )

st.divider()
st.subheader("Processed Files Log")

if os.path.exists("mvp3_dataframe.csv"):
    log_df = pd.read_csv("mvp3_dataframe.csv")
    if not log_df.empty:
        st.dataframe(log_df, width='stretch')

        st.download_button(
            label="Download Log as CSV",
            data=log_df.to_csv(index=False),
            file_name="mvp3_dataframe.csv",
            mime="text/csv",
        )

        if st.button("Clear Log"):
            os.remove("mvp3_dataframe.csv")
            st.rerun()
    else:
        st.info("No files processed yet.")
else:
    st.info("No files processed yet.")
