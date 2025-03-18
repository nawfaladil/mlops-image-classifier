FROM mysql:8.0

# Définir les variables d'environnement
ENV MYSQL_ROOT_PASSWORD=root
ENV MYSQL_DATABASE=mlops_db
ENV MYSQL_USER=mlops_user
ENV MYSQL_PASSWORD=mlops_pass

# Copier le script SQL pour initialiser la base de données
COPY init_db.sql /docker-entrypoint-initdb.d/

# Exposer le port MySQL
EXPOSE 3306