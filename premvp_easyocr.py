"""
Phase 1

This application demonstrates a proof-of-concept pipeline for converting
scanned biopharma manufacturing batch record PDFs into structured data.

Pipeline:
PDF → Image Conversion → OCR → Field Extraction → Structured JSON/CSV Output

Tools:
- Streamlit (UI)
- pdf2image (PDF processing)
- EasyOCR (text recognition)
- Regex-based field extraction

Goal:
Enable automated extraction of manufacturing data from executed batch records
for downstream review, analytics, and data warehouse ingestion.
"""

import streamlit as st
from pdf2image import convert_from_bytes
import easyocr
import numpy as np
import pandas as pd
import re
import json
import cv2

st.set_page_config(page_title="Batch Record OCR MVP", layout="wide")

st.title("Batch Record OCR MVP")
st.write("Upload a scanned PDF batch record to extract text and simple structured fields.")

@st.cache_resource
def load_ocr_reader():
    return easyocr.Reader(["en"], gpu=False)

def preprocess_image(pil_image):
    image = np.array(pil_image)

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    processed = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )

    return processed

def extract_fields(page_text: str) -> dict:
    fields = {
        "batch_number": None,
        "lot_number": None,
        "date": None,
        "operator": None,
    }

    patterns = {
        "batch_number": [
            r"Batch\s*(?:No|Number|Record)?[:\-]?\s*([A-Za-z0-9\-\/]+)",
        ],
        "lot_number": [
            r"Lot\s*(?:No|Number)?[:\-]?\s*([A-Za-z0-9\-\/]+)",
        ],
        "date": [
            r"Date[:\-]?\s*([0-9]{1,2}[\/\-][0-9]{1,2}[\/\-][0-9]{2,4})",
        ],
        "operator": [
            r"Operator[:\-]?\s*([A-Za-z]{1,20})",
            r"Initials[:\-]?\s*([A-Za-z]{1,10})",
        ],
    }

    for field_name, field_patterns in patterns.items():
        for pattern in field_patterns:
            match = re.search(pattern, page_text, re.IGNORECASE)
            if match:
                fields[field_name] = match.group(1).strip()
                break

    return fields

def build_ocr_entries(results: list) -> list:
    entries = []

    for item in results:
        bbox, text, confidence = item

        # Convert numpy coordinates to normal Python ints
        bbox_clean = [[int(x), int(y)] for x, y in bbox]

        entries.append({
            "text": str(text),
            "confidence": float(confidence),
            "bbox": bbox_clean
        })

    return entries

reader = load_ocr_reader()

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file is not None:
    pdf_bytes = uploaded_file.read()

    with st.spinner("Converting PDF pages to images..."):
        pages = convert_from_bytes(pdf_bytes, dpi=200)

    st.success(f"Converted {len(pages)} pages.")

    max_pages = st.slider(
        "Number of pages to process",
        min_value=0,
        max_value=len(pages),
        value=min(len(pages), 5)
    )

    all_results = []

    for i in range(max_pages):
        st.markdown("---")
        st.subheader(f"Page {i + 1}")

        original_image = pages[i]
        processed_image = preprocess_image(original_image)

        col1, col2 = st.columns(2)

        with col1:
            st.image(original_image, caption=f"Original Page {i + 1}", width='content')

        with st.spinner(f"Running OCR on page {i + 1}..."):
            results = reader.readtext(processed_image)

        page_text_lines = [item[1] for item in results]
        page_text = "\n".join(page_text_lines)

        extracted_fields = extract_fields(page_text)
        ocr_entries = build_ocr_entries(results)

        page_result = {
            "page_number": i + 1,
            "ocr_text": page_text,
            "fields": extracted_fields,
            "ocr_entries": ocr_entries,
        }
        all_results.append(page_result)

        with col2:
            st.write("### Extracted Fields")
            st.json(extracted_fields)

            st.write("### OCR Text")
            st.text_area(
                f"OCR Text - Page {i + 1}",
                page_text,
                height=300,
                key=f"text_{i}"
            )

            with st.expander("Show OCR Entries"):
                st.json(ocr_entries[:20])

    st.markdown("---")
    st.subheader("Extracted Fields Summary")

    summary_rows = []
    for page in all_results:
        row = {"page_number": page["page_number"]}
        row.update(page["fields"])
        summary_rows.append(row)

    df = pd.DataFrame(summary_rows)
    st.dataframe(df, width='stretch')

    csv_output = df.to_csv(index=False)

    st.download_button(
        label="Download CSV Summary",
        data=csv_output,
        file_name="batch_record_summary.csv",
        mime="text/csv",
    )

    st.subheader("Combined JSON Output")
    json_output = json.dumps(all_results, indent=2)

    #  json output is long
    # st.code(json_output, language="json")

    st.download_button(
        label="Download JSON Results",
        data=json_output,
        file_name="batch_record_results.json",
        mime="application/json",
    )