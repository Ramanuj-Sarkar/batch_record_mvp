"""
Phase 2: Layout-Aware Batch Record Extraction

This version extends the OCR MVP by introducing a document transformer
(such as Donut) to improve extraction from form-style biopharma batch
record pages where layout and field positioning are important.

Goal:
Improve structured field extraction accuracy and reduce dependence on
traditional OCR + regex rules for layout-sensitive documents.
"""
import streamlit as st
from pdf2image import convert_from_bytes
import easyocr
import numpy as np
import pandas as pd
import re
import cv2
import json
import torch
from PIL import Image
from transformers import DonutProcessor, VisionEncoderDecoderModel


st.set_page_config(page_title="Batch Record OCR MVP", layout="wide")

st.title("Batch Record OCR MVP")
st.write("Upload a scanned PDF batch record to extract text and simple structured fields.")


class DonutExtractor:
    """
    Wrapper around Donut model for structured document extraction.
    """
    def __init__(
        self,
        model_name: str = "naver-clova-ix/donut-base-finetuned-cord-v2",
        device: str | None = None,
    ):
        """
        Initialize model and processor.

        Args:
            model_name: pretrained Donut model
            device: cpu or cuda
        """

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.processor = DonutProcessor.from_pretrained(model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name)

        self.model.to(self.device)
        self.model.eval()

        self.task_prompt = "<s_cord-v2>"

    def preprocess_image(self, image: Image.Image):
        """
        Convert PIL image to model input tensor.
        """
        if image.mode != "RGB":
            image = image.convert("RGB")

        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        return pixel_values.to(self.device)

    def generate_prediction(self, pixel_values):
        """
        Run model inference.
        """

        decoder_input_ids = self.processor.tokenizer(
            self.task_prompt,
            add_special_tokens=False,
            return_tensors="pt",
        ).input_ids.to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                pixel_values,
                decoder_input_ids=decoder_input_ids,
                max_length=self.model.decoder.config.max_position_embeddings,
                pad_token_id=self.processor.tokenizer.pad_token_id,
                eos_token_id=self.processor.tokenizer.eos_token_id,
                use_cache=True,
                bad_words_ids=[[self.processor.tokenizer.unk_token_id]],
                return_dict_in_generate=True,
            )

        sequence = self.processor.batch_decode(outputs.sequences)[0]
        sequence = sequence.replace(self.processor.tokenizer.eos_token, "")
        sequence = sequence.replace(self.processor.tokenizer.pad_token, "")
        sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()

        return sequence

    def parse_output(self, model_output: str):
        """
        Convert model output string into JSON dictionary.
        """
        try:
            if hasattr(self.processor, "token2json"):
                return self.processor.token2json(model_output)
        except Exception:
            pass

        try:
            start = model_output.find("{")
            end = model_output.rfind("}") + 1
            if start != -1 and end > start:
                return json.loads(model_output[start:end])
        except Exception:
            pass

        return {
            "raw_output": model_output,
            "parse_error": True
        }

    def extract(self, image: Image.Image):
        """
        Main extraction function.

        Args:
            image: PIL image of page

        Returns:
            dict with structured fields
        """

        pixel_values = self.preprocess_image(image)

        model_output = self.generate_prediction(pixel_values)

        parsed = self.parse_output(model_output)

        return {
            "method": "donut",
            "prediction": parsed,
            "raw_output": model_output
        }


donut_extractor = DonutExtractor()


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

        donut_result = donut_extractor.extract(original_image)

        st.write("### Donut Extraction")
        st.json(donut_result["prediction"])

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