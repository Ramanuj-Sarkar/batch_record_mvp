import json
import os
import re
from collections import Counter

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from mvp3_azure_extractor import AzureDocIntExtractor

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
                r"Batch\s*No[:\-]?\s*([A-Za-z0-9\-\/]+)",
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
    ["prebuilt-read", "prebuilt-layout", "custom-model"],
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

                st.success("Document processed successfully.")

                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Normalized Batch Record Output")
                    st.json(normalized_result)

                with col2:
                    st.subheader("Raw Azure Output")
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
