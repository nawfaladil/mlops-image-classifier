# 🌿 MLOps Image Classifier Pipeline

## 🚀 Objectif

Développement d’un pipeline MLOps complet pour la classification d’images de plantes (`dandelion` vs `grass`) à l’aide de :

- **Airflow** pour l’orchestration
- **MySQL** pour les métadonnées
- **MinIO** comme stockage objet (S3 local)
- **Docker / Kubernetes** pour la conteneurisation et le déploiement (local)
- **MLflow** pour le suivi des modèles
- **FastAPI** & **Gradio** pour l’interface d’inférence
- **GitHub Actions** pour une intégration & livraison continues (CI/CD)

---

## 🧠 Étapes du projet

### 1. **Extraction et stockage des images**
- Récupération des URLs d’images depuis une base MySQL
- Téléchargement des images depuis un dépôt github
- Stockage dans un bucket S3 local (MinIO)
- Mise à jour des métadonnées dans la base

### 2. **Prétraitement des images**
- Redimensionnement et normalisation
- Stockage du dataset prétraité dans un autre bucket MinIO

### 3. **Entraînement du modèle**
- Chargement des données depuis MinIO
- Entraînement d’un modèle de classification
- Sauvegarde du modèle entraîné dans MinIO
- Logging des métriques et du modèle avec MLflow

### 4. **Déploiement du modèle**
- Création d’une API d’inférence avec FastAPI
- Interface utilisateur avec Gradio pour tester les prédictions
- Conteneurisation et exposition via Kubernetes

---

## 🔁 CI/CD avec GitHub Actions

- **Tests automatiques** à chaque `push` sur la branche main
- **Push automatique** de l'image fastapi sur Docker Hub

---

## ☸️ Déploiement avec Kubernetes

Le déploiement est fait avec l'image de fast api uploadée sur docker hub, le déploiement est donc mis à jour après le push automatique de github actions.

---

## Dév/production

Pour simuler un environement de développement et un environement de production séparés, nous avons travaillé avec deux branches différentes : une branch dev et une branch main pour la prod.
