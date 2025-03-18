USE mlops_db;

-- Drop procedure if it exists
DROP PROCEDURE IF EXISTS insert_images;

DELIMITER $$

CREATE PROCEDURE insert_images()
BEGIN
    DECLARE current INT DEFAULT 0;
    DECLARE end_val INT DEFAULT 199;  -- Set max image index

    -- Insert Dandelion images
    WHILE current <= end_val DO
        INSERT INTO plants_data (url_source, url_s3, label)
        VALUES (
            CONCAT('https://raw.githubusercontent.com/btphan95/greenr-airflow/refs/heads/master/data/dandelion/', LPAD(current, 8, '0'), '.jpg'),
            NULL,
            'dandelion'
        );
        SET current = current + 1;
    END WHILE;

    -- Reset counter for Grass images
    SET current = 0;

    -- Insert Grass images
    WHILE current <= end_val DO
        INSERT INTO plants_data (url_source, url_s3, label)
        VALUES (
            CONCAT('https://raw.githubusercontent.com/btphan95/greenr-airflow/refs/heads/master/data/grass/', LPAD(current, 8, '0'), '.jpg'),
            NULL,
            'grass'
        );
        SET current = current + 1;
    END WHILE;

END $$

DELIMITER ;

-- Execute the procedure to insert data
CALL insert_images();
