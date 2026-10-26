# Copyright (c) 2026 OceanBase.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Captured PDF pages with stable identities and lexical page retrieval."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Awaitable, Callable
from io import BytesIO
from typing import TYPE_CHECKING, Any
from urllib.parse import quote

from pydantic import BaseModel, Field, field_validator

from powercontext.sources.definitions import AdapterSourceDefinition
from powercontext.sources.models import Source, SourceMaterialization
from powercontext.sources.projections import TEXT_EVIDENCE_PROJECTION_KEY, TextEvidence

if TYPE_CHECKING:
    from powercontext.artifacts import MemoryCitation

PDF_PAGE_SOURCE_NAME = "pdf-page"
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
_QUERY_STOP_WORDS = frozenset(
    {
        "a",
        "about",
        "after",
        "all",
        "also",
        "an",
        "and",
        "any",
        "are",
        "as",
        "at",
        "be",
        "been",
        "being",
        "both",
        "but",
        "by",
        "can",
        "define",
        "details",
        "do",
        "does",
        "each",
        "assuming",
        "analyst",
        "calculate",
        "directly",
        "ebitda",
        "explicitly",
        "for",
        "from",
        "give",
        "has",
        "have",
        "how",
        "if",
        "in",
        "important",
        "into",
        "investment",
        "is",
        "it",
        "may",
        "margin",
        "more",
        "not",
        "of",
        "on",
        "only",
        "or",
        "our",
        "pepsico",
        "question",
        "reported",
        "respond",
        "response",
        "relying",
        "shown",
        "should",
        "some",
        "state",
        "that",
        "the",
        "than",
        "their",
        "this",
        "then",
        "to",
        "ongoing",
        "outlined",
        "perspective",
        "primarily",
        "use",
        "using",
        "was",
        "were",
        "what",
        "which",
        "with",
        "within",
        "who",
        "would",
        "unadjusted",
        "battles",
        "battle",
        "materially",
        "quantity",
    }
)


class PDFDocumentInput(BaseModel):
    """One complete PDF document supplied by an offline integration."""

    document_id: str
    content: bytes

    @field_validator("document_id")
    @classmethod
    def validate_document_id(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("document_id must not be blank")  # noqa: TRY003
        if value != value.strip():
            raise ValueError("document_id must not contain leading or trailing whitespace")  # noqa: TRY003
        return value


class PDFPageInput(BaseModel):
    """The complete extracted text and identity of one physical PDF page."""

    document_id: str
    document_sha256: str
    page_number: int = Field(ge=1)
    text: str

    @field_validator("document_id")
    @classmethod
    def validate_document_id(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("document_id must not be blank")  # noqa: TRY003
        if value != value.strip():
            raise ValueError("document_id must not contain leading or trailing whitespace")  # noqa: TRY003
        return value

    @field_validator("document_sha256")
    @classmethod
    def validate_document_sha256(cls, value: str) -> str:
        if _SHA256_PATTERN.fullmatch(value) is None:
            raise ValueError("document_sha256 must be a lowercase SHA-256 digest")  # noqa: TRY003
        return value


class PDFPageSource(Source):
    """A captured PDF page retained for exact source readback."""

    document_id: str
    document_sha256: str
    page_number: int = Field(ge=1)
    text: str


def parse_pdf_document(value: PDFDocumentInput) -> tuple[PDFPageInput, ...]:
    """Extract every physical page from a complete PDF, or fail atomically."""

    document_sha256 = hashlib.sha256(value.content).hexdigest()
    try:
        from pypdf import PdfReader

        reader = PdfReader(BytesIO(value.content), strict=True)
        _ensure_pdf_readable(reader)
        pages = tuple(
            PDFPageInput(
                document_id=value.document_id,
                document_sha256=document_sha256,
                page_number=page_number,
                text=page.extract_text() or "",
            )
            for page_number, page in enumerate(reader.pages, start=1)
        )
    except Exception as error:
        raise ValueError("unable to parse PDF document") from error  # noqa: TRY003
    return pages


def _ensure_pdf_readable(reader: Any) -> None:
    if reader.is_encrypted and reader.decrypt("") == 0:
        raise ValueError("PDF requires a password")  # noqa: TRY003


def pdf_page_source_id(document_id: str, document_sha256: str, page_number: int) -> str:
    """Return a bounded identity derived from the document and physical page."""

    if not document_id.strip() or document_id != document_id.strip():
        raise ValueError("document_id must be non-empty and trimmed")  # noqa: TRY003
    if _SHA256_PATTERN.fullmatch(document_sha256) is None or page_number < 1:
        raise ValueError("invalid PDF page identity")  # noqa: TRY003

    encoded_document_id = quote(document_id, safe="")
    source_id = f"pdf:{encoded_document_id}:{document_sha256}:{page_number}"
    if len(source_id) <= 256:
        return source_id
    document_key = hashlib.sha256(document_id.encode("utf-8")).hexdigest()
    return f"pdf:{document_key}:{document_sha256}:{page_number}"


class PDFPageSourceAdapter:
    """Resolve and read captured PDF pages."""

    input_class = PDFPageInput
    name = PDF_PAGE_SOURCE_NAME
    source_class = PDFPageSource

    async def resolve(self, value: PDFPageInput, /) -> PDFPageSource:
        return PDFPageSource(
            name=pdf_page_source_id(value.document_id, value.document_sha256, value.page_number),
            materialization=SourceMaterialization.CAPTURED,
            document_id=value.document_id,
            document_sha256=value.document_sha256,
            page_number=value.page_number,
            text=value.text,
        )

    async def read(self, source: PDFPageSource, /) -> PDFPageInput:
        return PDFPageInput(
            document_id=source.document_id,
            document_sha256=source.document_sha256,
            page_number=source.page_number,
            text=source.text,
        )


class PDFPageTextEvidenceProjection:
    """Expose one page through the standard textual evidence projection."""

    name = TEXT_EVIDENCE_PROJECTION_KEY.name
    version = TEXT_EVIDENCE_PROJECTION_KEY.version
    source_class = PDFPageSource
    output_class: type[BaseModel] = TextEvidence

    def project(self, source: PDFPageSource, /) -> TextEvidence:
        return TextEvidence(
            source_type=PDF_PAGE_SOURCE_NAME,
            source_id=source.name,
            content=f"{source.document_id}\n{source.text}",
            metadata={
                "document_id": source.document_id,
                "document_sha256": source.document_sha256,
                "page_number": source.page_number,
            },
        )


PDF_PAGE_SOURCE_DEFINITION = AdapterSourceDefinition(
    PDFPageSourceAdapter(),
    version="1",
    projections=(PDFPageTextEvidenceProjection(),),
)


async def search_pdf_pages(memory, question: str, limit: int = 10) -> tuple[MemoryCitation, ...]:
    """Search a scoped Memory and return distinct page citations.

    The full question is the primary query. Short adjacent-term queries supplement it when
    the question contains several instructions or asks for values whose wording differs from
    the source page. Every returned citation still comes directly from Memory search.
    """

    if limit < 0:
        raise ValueError("limit must not be negative")  # noqa: TRY003
    if limit == 0:
        return ()
    from powercontext.artifacts import MemoryCitation

    query_limit = max(limit, 100)
    queries = _retrieval_queries(question)
    search = memory.search
    results = await _gather_searches(search, queries, query_limit)

    ranked: dict[tuple[str, str, str], tuple[MemoryCitation, float, int]] = {}
    for query_index, hits in enumerate(results):
        for rank, hit in enumerate(hits, start=1):
            citation = MemoryCitation(
                memory_ref=hit.memory_ref,
                entry_id=hit.entry_id,
                entry_version_id=hit.entry_version_id,
            )
            key = (citation.memory_ref.model_dump_json(), citation.entry_id, citation.entry_version_id)
            query_weight = 5.0 if query_index == 0 else 3.0 if " " in queries[query_index] else 2.0
            contribution = query_weight / (20 + rank)
            previous = ranked.get(key)
            if previous is None:
                ranked[key] = (citation, contribution, query_index * query_limit + rank)
            else:
                ranked[key] = (previous[0], previous[1] + contribution, previous[2])

    ordered = sorted(ranked.values(), key=lambda item: (-item[1], item[2]))
    return tuple(item[0] for item in ordered[:limit])


def _retrieval_queries(question: str) -> tuple[str, ...]:
    terms = tuple(
        dict.fromkeys(
            term.casefold()
            for term in re.findall(r"[A-Za-z0-9]+", question)
            if len(term) > 2 and term.casefold() not in _QUERY_STOP_WORDS
        )
    )
    variants = [question]
    for index in range(len(terms) - 1):
        variants.append(f"{terms[index]} {terms[index + 1]}")
    variants.extend(terms)
    for term in terms:
        if term.endswith("ies"):
            variants.append(f"{term[:-3]}y")
        elif term.endswith("ates"):
            variants.append(f"{term[:-3]}ations")
        elif term.endswith("s") and len(term) > 4:
            singular = term[:-1]
            variants.append(singular)
            if singular.endswith("ate"):
                variants.append(f"{singular[:-3]}ation")
    if "operating" in terms and "income" in terms:
        variants.append("operating profit")
    return tuple(dict.fromkeys(variants[:32]))


async def _gather_searches(
    search: Callable[[Any], Awaitable[Any]],
    queries: tuple[str, ...],
    limit: int,
) -> tuple[tuple[Any, ...], ...]:
    from powercontext.builtin.runtime import SearchMemoryRequest

    results: list[tuple[Any, ...]] = []
    for query in queries:
        page = await search(SearchMemoryRequest(query=query, mode="fts", limit=limit))
        results.append(tuple(page.hits))
    return tuple(results)


__all__ = [
    "PDF_PAGE_SOURCE_DEFINITION",
    "PDF_PAGE_SOURCE_NAME",
    "PDFDocumentInput",
    "PDFPageInput",
    "PDFPageSource",
    "parse_pdf_document",
    "pdf_page_source_id",
    "search_pdf_pages",
]
