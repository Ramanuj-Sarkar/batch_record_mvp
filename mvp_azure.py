import json
import os
import re
from collections import Counter

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from mvp_azure_extractor import AzureDocIntExtractor

load_dotenv()

st.set_page_config(page_title="Azure Batch Record Extraction MVP", layout="wide")

st.title("Biopharma Batch Record Extraction - Azure Version")
st.write(
    "Upload a PDF batch record and process it with Azure Document Intelligence. "
    "This version supports prebuilt OCR/layout analysis and custom model extraction."
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


def polygon_to_bbox(polygon):
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


def word_center_in_bbox(word_polygon, cell_bbox):
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
        page_number = region.get("page_number")
        polygon = region.get("polygon")
        cell_bbox = polygon_to_bbox(polygon)

        if not cell_bbox:
            continue

        matching_page = next(
            (p for p in pages if p.get("page_number") == page_number),
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
    headers = {}
    rows = {}

    for cell in t.get("cells", []):
        r = cell["row_index"]
        c = cell["column_index"]

        text = (cell.get("content") or "").strip()

        if not text:
            text = recover_cell_text_from_words(cell, pages)

        if r == 0:
            headers[c] = text if text else f"col_{c}"
        else:
            if r not in rows:
                rows[r] = {}

            column_name = headers.get(c, f"col_{c}")
            rows[r][column_name] = text

    return list(rows.values())


def parse_all_azure_tables(rr: dict) -> list[dict]:
    """
    Parse all Azure tables into structured row-based output.
    """
    parsed_tables = []
    pages = rr.get("pages", [])

    for idx, t in enumerate(rr.get("tables", []), start=1):
        parsed_rows = parse_azure_table(t, pages)

        parsed_tables.append({
            "table_index": idx,
            "row_count": t.get("row_count"),
            "column_count": t.get("column_count"),
            "rows": parsed_rows,
        })

    return parsed_tables


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

    for page in result.get("pages", []):
        page_number = page["page_number"]
        page_text = "\n".join(line["content"] for line in page.get("lines", []))

        batch_number = extract_with_patterns(
            page_text,
            [
                r"Batch\s*Number[:\-]?\s*([A-Za-z0-9\-\/]+)",
                r"Batch\s*Record\s*:\n*\s*([A-Za-z0-9\-\/]+)"
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

        normalized["pages"].append(
            {
                "page_number": page_number,
                "page_type": "unknown_page",
                "ocr_text": page_text,
                "fields": {
                    "batch_number": batch_number,
                    "lot_number": lot_number,
                    "date": date_value,
                    "operator": operator,
                },
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

    return normalized


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
                    field_obj = fields[name]
                    for key in ["value_string", "value_date", "value_time", "content", "value_number"]:
                        value = field_obj.get(key)
                        if value is not None:
                            return str(value)
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
            }
        )

    return normalized


def build_summary_dataframe(normalized: dict) -> pd.DataFrame:
    rows = []
    for page in normalized.get("pages", []):
        row = {
            "page_number": page.get("page_number"),
            "page_type": page.get("page_type"),
        }
        row.update(page.get("fields", {}))
        rows.append(row)
    return pd.DataFrame(rows)


azure_extractor = load_azure_extractor()

with st.sidebar:
    st.header("Azure Settings")
    endpoint_present = bool(os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT"))
    key_present = bool(os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY"))
    st.write(f"Endpoint loaded: {'Yes' if endpoint_present else 'No'}")
    st.write(f"Key loaded: {'Yes' if key_present else 'No'}")

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

model_choice = st.selectbox(
    "Choose Azure model",
    ["prebuilt-layout", "prebuilt-read", "custom-model"],
)

pages_to_process = st.text_input(
    "Pages to process (optional)",
    placeholder="Example: 1-3 or 1,3,5",
)

default_custom_model_id = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_CUSTOM_MODEL_ID", "")
custom_model_id = ""

if model_choice == "custom-model":
    custom_model_id = st.text_input(
        "Custom model ID",
        value=default_custom_model_id,
    )

if uploaded_file is not None:
    pdf_bytes = uploaded_file.read()

    if st.button("Process Document"):
        with st.spinner("Sending document to Azure Document Intelligence..."):
            try:
                if model_choice == "prebuilt-read":
                    raw_result = azure_extractor.analyze_read(
                        pdf_bytes,
                        pages=pages_to_process or None,
                    )
                    normalized_result = normalize_prebuilt_result(raw_result)

                elif model_choice == "prebuilt-layout":
                    raw_result = azure_extractor.analyze_layout(
                        pdf_bytes,
                        pages=pages_to_process or None,
                    )
                    normalized_result = normalize_prebuilt_result(raw_result)

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

                col1, col2 = st.columns(2)

                with col1:
                    with st.expander("Show Normalized Batch Record Output"):
                        st.json(normalized_result)

                with col2:
                    with st.expander("Show Raw Azure Output"):
                        st.json(raw_result)

                if normalized_result.get("validation_warnings"):
                    st.subheader("Validation Warnings")
                    for warning in normalized_result["validation_warnings"]:
                        st.warning(warning)

                st.subheader("Summary Table")
                df = build_summary_dataframe(normalized_result)
                st.dataframe(df, use_container_width=True)

                normalized_json = json.dumps(normalized_result, indent=2)
                raw_json = json.dumps(raw_result, indent=2)

                if parsed_tables:
                    st.subheader("Parsed Azure Tables")

                    for table in parsed_tables:
                        st.write(
                            f"Table {table['table_index']} "
                            f"({table['row_count']} rows x {table['column_count']} columns)"
                        )

                        table_rows = table["rows"]
                        if table_rows:
                            table_df = pd.DataFrame(table_rows)
                            st.dataframe(table_df, use_container_width=True)
                        else:
                            st.info("No parsed rows found for this table.")

                st.download_button(
                    label="Download Normalized JSON",
                    data=normalized_json,
                    file_name="batch_record_normalized.json",
                    mime="application/json",
                )

                st.download_button(
                    label="Download Raw Azure JSON",
                    data=raw_json,
                    file_name="batch_record_raw_azure.json",
                    mime="application/json",
                )

                csv_output = df.to_csv(index=False)
                st.download_button(
                    label="Download CSV Summary",
                    data=csv_output,
                    file_name="batch_record_summary.csv",
                    mime="text/csv",
                )

            except Exception as e:
                st.error(f"Error processing document: {e}")
