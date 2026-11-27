# Movie Recommender System

Movie recommendation service built on the MovieLens 100K dataset, centered around a **Two-Tower Retrieval (TTR)** neural model, with SVD as a baseline, FastAPI, Streamlit, and Docker.

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?logo=fastapi)
![Docker](https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)


## Overview

This project is a movie recommendation  app built on the MovieLens 100K dataset. The core idea of the project is implementing a **Two-Tower Retrieval (TTR)** model, where users and movies are encoded by two separate neural networks (Towers) into a shared embedding space to retrieve relevant candidates based on embedding similarity.

To set a baseline and build a working end-to-end service first, I started with collaborative filtering (**SVD**), built a REST API with **FastAPI** and a UI with **Streamlit**, and packaged everything in **Docker**.

### Dataset & Modeling Summary
* **Dataset:** [MovieLens 100K](https://grouplens.org)
* **Matrix Sparsity:** **6.3%** filled (`100,000` ratings out of `1,586,126` possible user-movie pairs)
* **Core Model:** Two-Tower Retrieval (TTR) *(main focus, in active development)*
* **Baseline Model:** Collaborative filtering with **SVD** (`scikit-surprise`) — **RMSE ≈ 1.021**

---

## Project Structure

```text
movie-recsys/
├── two-tower-retrieval/    # Core Two-Tower Retrieval (TTR) model & training scripts
├── api/                    # FastAPI app, endpoints, and Pydantic schemas
├── frontend/               # Streamlit web interface
├── models/                 # Saved model files and weights
├── notebooks/              # Data exploration and experiments
├── src/                    # Data preprocessing and SVD baseline pipeline
├── docker/                 # Dockerfiles for API and frontend
├── docs/                   # Project notes and documentation
├── docker-compose.yml      # Docker Compose setup
└── requirements.txt        # Python dependencies
```

---

## Tech Stack

| Part | Tools |
| :--- | :--- |
| **Core Model & ML** | `Python 3.11`, `PyTorch` (Two-Tower Retrieval), `NumPy`, `Pandas`, `scikit-learn`, `scikit-surprise` (SVD baseline) |
| **Backend** | `FastAPI`, `Pydantic`, `Uvicorn` |
| **Frontend** | `Streamlit` |
| **Tools & Deployment** | `Docker`, `Docker Compose`|

---

## Quick Start

### Run with Docker Compose (Recommended)
If you have **Docker** and **Docker Compose** installed, start both services with one command:

```bash
git clone 
cd movie-recsys
docker-compose up --build
```

Once running, open in your browser:
* **Streamlit App:** `http://localhost:8501`
* **FastAPI Docs:** `http://localhost:8000/docs`

### Local Setup
```bash
git clone https://github.com/Ivampypy/movie-recsys.git
cd movie-recsys
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## Roadmap

- [x] Data exploration and user-item matrix analysis
- [x] Baseline collaborative filtering model using **SVD** (`RMSE ≈ 1.021`)
- [x] REST API with **FastAPI** and **Pydantic** validation
- [x] Web interface with **Streamlit**
- [x] Docker and Docker Compose setup
- [x] **Two-Tower Retrieval (TTR):** Completing model training, retrieval evaluation (`Recall@K` / `HitRate@K`)
- [x] Offline Benchmark: Two-Tower vs SVD (`Recall@10`, `NDCG@10`)
- [x] FAISS Vector Index (`IndexFlatIP`) for Sub-Millisecond Candidate Retrieval
- [x] FastAPI Inference Endpoints (`/recommend/two-tower/{user_id}` & `/recommend/cold-start`)
- [x] Streamlit UI Update (Multi-page: SVD + Two-Tower Cold-Start) & Docker Compose Deployment

---
## Benchmark Results (Temporal Split 80/20)

Both models were tested on the same chronological split (first 80% of user ratings for training, last 20% for testing) across all 1,682 movies:

| Model | Recall@10 | NDCG@10 | Key Strength |
| :--- | :---: | :---: | :--- |
| **SVD (Baseline)** | `0.0456` | **`0.0662`** | Strong at ranking known items for active users  |
| **Two-Tower (HF + MLP)** | **`0.0576`** (+26.3%) | `0.0601` | Better candidate recall and supports cold-start users  |

## Limitations

* **Small Dataset (MovieLens 100K):** The dataset only has 100K ratings (943 users and 1,682 movies). It is good enough to build and test the project, though neural models like Two-Tower Retrieval usually work much better on larger datasets.
- **No Automatic Retraining:** Models are trained offline on static data. If new ratings are added, the model does not update automatically.

## Contacts
Any suggestions/questions? You can keep in touch with me via:
- Telegram: **@nepipipip**
- LinkedIn: **www.linkedin.com/in/ivan-karzin-ml**
