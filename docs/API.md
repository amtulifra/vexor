# Vexor HTTP API

Run the API server:

```bash
uvicorn vexor.api.app:app --host 0.0.0.0 --port 8000
```

## Health

```bash
curl http://127.0.0.1:8000/health
```

## Create an index

```bash
curl -X POST http://127.0.0.1:8000/indexes \
  -H "content-type: application/json" \
  -d '{
    "name": "demo",
    "dim": 8,
    "index_type": "flat",
    "metric": "l2"
  }'
```

## Add vectors (batch)

```bash
curl -X POST http://127.0.0.1:8000/indexes/demo/vectors/batch \
  -H "content-type: application/json" \
  -d '{
    "vectors": [[0,1,0,1,0,1,0,1], [1,0,1,0,1,0,1,0]]
  }'
```

## Queue background ingestion

```bash
curl -X POST http://127.0.0.1:8000/indexes/demo/ingest/queue \
  -H "content-type: application/json" \
  -d '{
    "vectors": [[0,1,0,1,0,1,0,1], [1,0,1,0,1,0,1,0]]
  }'
```

Then inspect job status:

```bash
curl http://127.0.0.1:8000/indexes/demo/ingest/jobs/<job_id>
```

Force drain the queue (useful in tests/dev):

```bash
curl -X POST "http://127.0.0.1:8000/indexes/demo/ingest/flush?timeout_s=5"
```

## Search (single)

```bash
curl -X POST http://127.0.0.1:8000/indexes/demo/search \
  -H "content-type: application/json" \
  -d '{
    "query": [0,1,0,1,0,1,0,1],
    "k": 2
  }'
```

## Search (batch)

```bash
curl -X POST http://127.0.0.1:8000/indexes/demo/search/batch \
  -H "content-type: application/json" \
  -d '{
    "queries": [[0,1,0,1,0,1,0,1], [1,0,1,0,1,0,1,0]],
    "k": 2
  }'
```

## Save and load

```bash
curl -X POST http://127.0.0.1:8000/indexes/demo/save \
  -H "content-type: application/json" \
  -d '{"path":"./demo.snap"}'
```

```bash
curl -X POST http://127.0.0.1:8000/indexes/load \
  -H "content-type: application/json" \
  -d '{"name":"demo2","path":"./demo.snap"}'
```
