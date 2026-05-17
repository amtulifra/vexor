"""FastAPI application exposing VectorDB operations over HTTP."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException

from vexor.api.schemas import (
    AddBatchRequest,
    AddVectorRequest,
    CreateIndexRequest,
    HealthResponse,
    IngestJobResponse,
    IngestQueueRequest,
    IngestQueueResponse,
    IndexInfo,
    LoadIndexRequest,
    SaveRequest,
    SearchBatchRequest,
    SearchBatchResponse,
    SearchHit,
    SearchRequest,
    SearchResponse,
    TrainRequest,
)
from vexor.api.service import IndexRegistry


def create_app() -> FastAPI:
    app = FastAPI(title="Vexor API", version="0.1.0")
    registry = IndexRegistry()

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse()

    @app.post("/indexes", response_model=IndexInfo)
    def create_index(req: CreateIndexRequest) -> IndexInfo:
        try:
            db = registry.create(
                name=req.name,
                dim=req.dim,
                index_type=req.index_type,
                metric=req.metric,
                wal_path=req.wal_path,
                index_kwargs=req.index_kwargs,
            )
            return IndexInfo(name=req.name, index_type=db.index_type, dim=db.dim, size=db.size)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/indexes", response_model=list[IndexInfo])
    def list_indexes() -> list[IndexInfo]:
        managed = registry.list()
        return [
            IndexInfo(name=m.name, index_type=m.db.index_type, dim=m.db.dim, size=m.db.size)
            for m in managed
        ]

    @app.delete("/indexes/{name}")
    def delete_index(name: str) -> dict[str, str]:
        try:
            registry.delete(name)
            return {"status": "deleted"}
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/indexes/load", response_model=IndexInfo)
    def load_index(req: LoadIndexRequest) -> IndexInfo:
        try:
            db = registry.load(name=req.name, path=req.path, wal_path=req.wal_path)
            return IndexInfo(name=req.name, index_type=db.index_type, dim=db.dim, size=db.size)
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/indexes/{name}/train")
    def train_index(name: str, req: TrainRequest) -> dict[str, str]:
        try:
            matrix = registry.to_array_2d(req.vectors)
            registry.train(name, matrix, **req.kwargs)
            return {"status": "trained"}
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (ValueError, RuntimeError, NotImplementedError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/indexes/{name}/vectors")
    def add_vector(name: str, req: AddVectorRequest) -> dict[str, int]:
        try:
            vec = registry.to_array_1d(req.vector)
            vec_id = registry.add_vector(name, vec, metadata=req.metadata)
            return {"vec_id": vec_id}
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/indexes/{name}/vectors/batch")
    def add_vectors_batch(name: str, req: AddBatchRequest) -> dict[str, list[int]]:
        try:
            matrix = registry.to_array_2d(req.vectors)
            ids = registry.add_batch(name, matrix, metadata=req.metadata)
            return {"vec_ids": ids}
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/indexes/{name}/ingest/queue", response_model=IngestQueueResponse)
    def queue_ingestion(name: str, req: IngestQueueRequest) -> IngestQueueResponse:
        try:
            matrix = registry.to_array_2d(req.vectors)
            job_id = registry.enqueue_ingestion_batch(name, matrix, metadata=req.metadata)
            return IngestQueueResponse(job_id=job_id, status="queued")
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/indexes/{name}/ingest/jobs/{job_id}", response_model=IngestJobResponse)
    def ingestion_status(name: str, job_id: str) -> IngestJobResponse:
        try:
            info = registry.ingestion_job_status(name, job_id)
            return IngestJobResponse(
                job_id=info["job_id"],
                status=info["status"],
                count=int(info["count"]),
                inserted=int(info["inserted"]),
                error=info["error"],
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/indexes/{name}/ingest/flush")
    def ingestion_flush(name: str, timeout_s: float = 5.0) -> dict[str, bool]:
        try:
            return {"drained": registry.flush_ingestion(name, timeout_s=timeout_s)}
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.delete("/indexes/{name}/vectors/{vec_id}")
    def delete_vector(name: str, vec_id: int) -> dict[str, str]:
        try:
            registry.delete_vector(name, vec_id)
            return {"status": "deleted"}
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except NotImplementedError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/indexes/{name}/search", response_model=SearchResponse)
    def search(name: str, req: SearchRequest) -> SearchResponse:
        try:
            query = registry.to_array_1d(req.query)
            hits = registry.search(name, query, k=req.k, filter=req.filter, **req.search_kwargs)
            return SearchResponse(results=[SearchHit(vec_id=vid, distance=dist) for vid, dist in hits])
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/indexes/{name}/search/batch", response_model=SearchBatchResponse)
    def search_batch(name: str, req: SearchBatchRequest) -> SearchBatchResponse:
        try:
            matrix = registry.to_array_2d(req.queries)
            results = registry.search_batch(name, matrix, k=req.k, filter=req.filter, **req.search_kwargs)
            return SearchBatchResponse(
                results=[
                    [SearchHit(vec_id=vid, distance=dist) for vid, dist in per_query]
                    for per_query in results
                ]
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/indexes/{name}/save")
    def save_index(name: str, req: SaveRequest) -> dict[str, str]:
        try:
            registry.save(name, req.path)
            return {"status": "saved"}
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return app


app = create_app()
