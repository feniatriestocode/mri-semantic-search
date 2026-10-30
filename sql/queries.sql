-- This file is written for reference and does not make or brake the benchmark, 
-- as SQL is embedded in the python code. 
-- It is here to help the reader understand the queries used in the benchmark.
-- It sets off the queries for Exact nearest Neighbor Search (Exact NNS) 
-- and Approximate Nearest Neighbor Search (Approximate NNS) for the benchmark.

-- ENN
SET enable_indexscan = off; -- No index available! The optimizer is forced not to use one

EXPLAIN (ANALYZE, BUFFERS)
SELECT id , image_name, category, (embedding <=> %s::vector) AS distance
FROM image_items
ORDER BY distance ASC
LIMIT 5; -- This is how you define how many nearest neighbors you want 

-- HNSW
SET enable_indexscan = on; -- Index available! The optimizer is free to use one
SET hnsw.ef_search = 40; 
EXPLAIN (ANALYZE, BUFFERS)
SELECT id , image_name, category, (embedding <=> %s::vector) AS distance
ORDER BY distance ASC
LIMIT 5; -- This is how you define how many nearest neighbors you want

-- IVFFLAT
SET enable_indexscan = on; 
SET ivfflat.probes = 5;
EXPLAIN (ANALYZE, BUFFERS)
SELECT id , image_name, category, (embedding <=> %s::vector) AS distance
ORDER BY distance ASC
LIMIT 5;