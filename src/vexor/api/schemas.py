"""Pydantic schemas for the Vexor HTTP API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"


class CreateIndexRequest(BaseModel):
    name: str = Field(..., min_length=1)
    dim: int = Field(..., ge=1)
    index_type: str = "hnsw"
    metric: str = "cosine"
    wal_path: str | None = None
    index_kwargs: dict[str, Any] = Field(default_factory=dict)


class IndexInfo(BaseModel):
    name: str
    index_type: str
    dim: int
    size: int


class AddVectorRequest(BaseModel):
    vector: list[float]
    metadata: dict[str, Any] | None = None


class AddBatchRequest(BaseModel):
    vectors: list[list[float]]
    metadata: list[dict[str, Any]] | None = None


class TrainRequest(BaseModel):
    vectors: list[list[float]]
    kwargs: dict[str, Any] = Field(default_factory=dict)


class SearchRequest(BaseModel):
    query: list[float]
    k: int = Field(10, ge=1)
    filter: dict[str, Any] | None = None
    search_kwargs: dict[str, Any] = Field(default_factory=dict)


class SearchBatchRequest(BaseModel):
    queries: list[list[float]]
    k: int = Field(10, ge=1)
    filter: dict[str, Any] | None = None
    search_kwargs: dict[str, Any] = Field(default_factory=dict)


class SearchHit(BaseModel):
    vec_id: int
    distance: float


class SearchResponse(BaseModel):
    results: list[SearchHit]


class SearchBatchResponse(BaseModel):
    results: list[list[SearchHit]]


class SaveRequest(BaseModel):
    path: str


class LoadIndexRequest(BaseModel):
    name: str = Field(..., min_length=1)
    path: str
    wal_path: str | None = None


class IngestQueueRequest(BaseModel):
    vectors: list[list[float]]
    metadata: list[dict[str, Any]] | None = None


class IngestQueueResponse(BaseModel):
    job_id: str
    status: str


class IngestJobResponse(BaseModel):
    job_id: str
    status: str
    count: int
    inserted: int
    error: str | None = None
