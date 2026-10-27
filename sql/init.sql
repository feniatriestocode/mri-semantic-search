CREATE EXTENSION IF NOT EXISTS vector; -- add the vector extension to support vector data types for image embeddings!

DROP TABLE IF EXISTS image_items CASCADE; -- drop the table if it already exists to avoid conflicts

CREATE TABLE image_items (
    id SERIAL PRIMARY KEY, -- unique identifier for each image
    image_name VARCHAR(255) NOT NULL, --filename of the image
    category VARCHAR(100) NOT NULL, -- class of the image
    embedding vector(512) NOT NULL, -- 512-dimensional vector representation of the image extracted by bioMedCLIP
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP -- timestamp of when the image was added to the database
);