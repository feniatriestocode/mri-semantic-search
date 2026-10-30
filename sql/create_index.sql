-- File that showcases the creation of the 2 ANN indices available in PGVector: ivfflat and hnsw. 
-- Note that the file itself is not vital to the code running, as SQL is injected into the database via the Python script. 
-- (benchmark.py). However, I decided to keep it here for reference purposes :)

-- The creation of the indexes here showcases the different parameters that can be used to create the indices.
-- I use cosine distance as the distance metric for both indices, but someone could use other metrics such as L2 or inner product.

-- IVFFLAT
CREATE INDEX items_ivfflat -- name of index
ON image_items  -- table to create the index on
USING ivfflat (embedding vector_cosine_ops) 
WITH (lists = 100); 

-- Delete index so that the optimizer can choose the other one when one has already ran
DROP INDEX IF EXISTS items_ivfflat;

--HSNW
CREATE INDEX items_hnsw -- name of index
ON image_items  -- table to create the index on
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64); -- m = links, ef_construction = number of candidates to consider 

-- Delete index so that the optimizer can choose the other one when one has already ran
DROP INDEX IF EXISTS items_hnsw;