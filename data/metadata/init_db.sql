CREATE DATABASE IF NOT EXISTS mlops_db;
USE mlops_db;

CREATE TABLE IF NOT EXISTS plants_data (
    id INT AUTO_INCREMENT PRIMARY KEY,
    url_source VARCHAR(255) NOT NULL,
    url_s3 VARCHAR(255),  -- Ce champ sera mis à jour après l’upload vers S3
    label ENUM('dandelion', 'grass') NOT NULL
);