"""
Extracts data from Azure Document Intelligence
to feed into MVP 2.
"""
import os
from typing import Any

from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from pypdf import PdfReader
from io import BytesIO


class AzureDocIntExtractor:
    def __init__(self, endpoint: str | None = None, key: str | None = None):
        self.endpoint = endpoint or os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
        self.key = key or os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY")

        if not self.endpoint or not self.key:
            raise ValueError(
                "Missing Azure Document Intelligence endpoint/key. "
                "Set AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT and "
                "AZURE_DOCUMENT_INTELLIGENCE_KEY."
            )

        self.client = DocumentIntelligenceClient(
            endpoint=self.endpoint,
            credential=AzureKeyCredential(self.key),
        )

    def get_pdf_page_count(self, pdf_bytes: bytes) -> int:
        reader = PdfReader(BytesIO(pdf_bytes))
        return len(reader.pages)

    def build_page_ranges(self, total_pages: int, chunk_size: int = 50) -> list[str]:
        ranges = []
        start = 1

        while start <= total_pages:
            end = min(start + chunk_size - 1, total_pages)
            ranges.append(f"{start}-{end}")
            start = end + 1

        return ranges

    def analyze_read(self, pdf_bytes: bytes, pages: str | None = None) -> dict[str, Any]:
        poller = self.client.begin_analyze_document(
            model_id="prebuilt-read",
            body=pdf_bytes,
            pages=pages,
        )
        result = poller.result()
        return self._read_result_to_dict(result)

    def analyze_layout(self, pdf_bytes: bytes, pages: str | None = None) -> dict[str, Any]:
        poller = self.client.begin_analyze_document(
            model_id="prebuilt-layout",
            body=pdf_bytes,
            pages=pages,
        )
        result = poller.result()
        return self._layout_result_to_dict(result)

    def analyze_custom(
        self,
        pdf_bytes: bytes,
        model_id: str,
        pages: str | None = None,
    ) -> dict[str, Any]:
        if not model_id:
            raise ValueError("Custom model ID is required for custom extraction.")

        poller = self.client.begin_analyze_document(
            model_id=model_id,
            body=pdf_bytes,
            pages=pages,
        )
        result = poller.result()
        return self._custom_result_to_dict(result)

    def analyze_read_chunked(
        self,
        pdf_bytes: bytes,
        chunk_size: int = 50,
    ) -> dict[str, Any]:
        total_pages = self.get_pdf_page_count(pdf_bytes)
        page_ranges = self.build_page_ranges(total_pages, chunk_size)

        chunk_results = []
        for page_range in page_ranges:
            chunk_results.append(self.analyze_read(pdf_bytes, pages=page_range))

        return self._merge_read_like_results(chunk_results, method="azure_prebuilt_read_chunked")

    def analyze_layout_chunked(
        self,
        pdf_bytes: bytes,
        chunk_size: int = 50,
    ) -> dict[str, Any]:
        total_pages = self.get_pdf_page_count(pdf_bytes)
        page_ranges = self.build_page_ranges(total_pages, chunk_size)

        chunk_results = []
        for page_range in page_ranges:
            chunk_results.append(self.analyze_layout(pdf_bytes, pages=page_range))

        return self._merge_layout_results(chunk_results, method="azure_prebuilt_layout_chunked")

    def analyze_custom_chunked(
        self,
        pdf_bytes: bytes,
        model_id: str,
        chunk_size: int = 50,
    ) -> dict[str, Any]:
        total_pages = self.get_pdf_page_count(pdf_bytes)
        page_ranges = self.build_page_ranges(total_pages, chunk_size)

        chunk_results = []
        for page_range in page_ranges:
            chunk_results.append(self.analyze_custom(pdf_bytes, model_id=model_id, pages=page_range))

        return self._merge_custom_results(chunk_results, method="azure_custom_extraction_chunked")

    def _merge_read_like_results(self, chunk_results: list[dict], method: str) -> dict[str, Any]:
        merged_pages = []

        for result in chunk_results:
            merged_pages.extend(result.get("pages", []))

        merged_pages.sort(key=lambda p: p["page_number"])

        return {
            "method": method,
            "pages": merged_pages,
        }

    def _merge_layout_results(self, chunk_results: list[dict], method: str) -> dict[str, Any]:
        merged = self._merge_read_like_results(chunk_results, method=method)
        merged["tables"] = []
        merged["paragraphs"] = []
        merged["styles"] = []

        for result in chunk_results:
            merged["tables"].extend(result.get("tables", []))
            merged["paragraphs"].extend(result.get("paragraphs", []))
            merged["styles"].extend(result.get("styles", []))

        return merged

    def _merge_custom_results(self, chunk_results: list[dict], method: str) -> dict[str, Any]:
        merged_documents = []

        for result in chunk_results:
            merged_documents.extend(result.get("documents", []))

        return {
            "method": method,
            "documents": merged_documents,
        }

    def _read_result_to_dict(self, result) -> dict[str, Any]:
        pages_out = []

        for page in result.pages:
            page_dict = {
                "page_number": int(page.page_number),
                "width": float(page.width) if page.width is not None else None,
                "height": float(page.height) if page.height is not None else None,
                "unit": str(page.unit) if page.unit is not None else None,
                "lines": [],
                "words": [],
            }

            if page.lines:
                for line in page.lines:
                    page_dict["lines"].append(
                        {
                            "content": line.content,
                            "polygon": self._normalize_polygon(line.polygon),
                        }
                    )

            if page.words:
                for word in page.words:
                    page_dict["words"].append(
                        {
                            "content": word.content,
                            "confidence": (
                                float(word.confidence)
                                if word.confidence is not None
                                else None
                            ),
                            "polygon": self._normalize_polygon(word.polygon),
                        }
                    )

            pages_out.append(page_dict)

        return {
            "method": "azure_prebuilt_read",
            "pages": pages_out,
        }

    def _layout_result_to_dict(self, result) -> dict[str, Any]:
        out = self._read_result_to_dict(result)
        out["method"] = "azure_prebuilt_layout"
        out["tables"] = []
        out["paragraphs"] = []
        out["styles"] = []

        if getattr(result, "paragraphs", None):
            for paragraph in result.paragraphs:
                out["paragraphs"].append(
                    {
                        "content": paragraph.content,
                        "bounding_regions": self._normalize_bounding_regions(
                            getattr(paragraph, "bounding_regions", None)
                        ),
                    }
                )

        if getattr(result, "styles", None):
            for style in result.styles:
                out["styles"].append(
                    {
                        "is_handwritten": getattr(style, "is_handwritten", None),
                        "confidence": (
                            float(style.confidence)
                            if getattr(style, "confidence", None) is not None
                            else None
                        ),
                    }
                )

        if result.tables:
            for table in result.tables:
                out["tables"].append(
                    {
                        "row_count": int(table.row_count),
                        "column_count": int(table.column_count),
                        "bounding_regions": self._normalize_bounding_regions(
                            getattr(table, "bounding_regions", None)
                        ),
                        "cells": [
                            {
                                "row_index": int(cell.row_index),
                                "column_index": int(cell.column_index),
                                "content": cell.content,
                                "kind": getattr(cell, "kind", None),
                                "bounding_regions": self._normalize_bounding_regions(
                                    getattr(cell, "bounding_regions", None)
                                ),
                            }
                            for cell in table.cells
                        ],
                    }
                )

        return out

    def _custom_result_to_dict(self, result) -> dict[str, Any]:
        documents = []

        if result.documents:
            for doc in result.documents:
                fields = {}

                if doc.fields:
                    for name, field in doc.fields.items():
                        fields[name] = {
                            "value_string": getattr(field, "value_string", None),
                            "value_number": getattr(field, "value_number", None),
                            "value_date": (
                                str(getattr(field, "value_date", None))
                                if getattr(field, "value_date", None)
                                else None
                            ),
                            "value_time": (
                                str(getattr(field, "value_time", None))
                                if getattr(field, "value_time", None)
                                else None
                            ),
                            "content": getattr(field, "content", None),
                            "confidence": (
                                float(field.confidence)
                                if getattr(field, "confidence", None) is not None
                                else None
                            ),
                        }

                documents.append(
                    {
                        "doc_type": doc.doc_type,
                        "confidence": (
                            float(doc.confidence)
                            if doc.confidence is not None
                            else None
                        ),
                        "fields": fields,
                    }
                )

        return {
            "method": "azure_custom_extraction",
            "documents": documents,
        }

    @staticmethod
    def _normalize_polygon(polygon):
        if not polygon:
            return None
        return [float(x) for x in polygon]

    @staticmethod
    def _normalize_bounding_regions(regions):
        if not regions:
            return []
        output = []
        for region in regions:
            output.append(
                {
                    "page_number": int(region.page_number),
                    "polygon": [float(x) for x in region.polygon] if region.polygon else None,
                }
            )
        return output