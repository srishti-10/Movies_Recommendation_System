# Movie Recommender System

A modular movie recommender system featuring collaborative filtering, content-based, and hybrid recommendation methods. The app is built with Streamlit for interactive exploration and evaluation.

## Features
- Collaborative Filtering using a custom neural network (PyTorch/Tez)
- Content-Based Filtering using TF-IDF and cosine similarity
- Hybrid Recommendations (placeholder for future extension)
- Interactive web UI with Streamlit
- Evaluation utilities for model performance

## Datasets
The following datasets are used (all should be placed in the `datasets/` directory):

- [`movies_metadata.csv`](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset): Movie metadata (title, genres, overview, etc.)
- [`keywords.csv`](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset): Keywords for movies
- [`links_small.csv`](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset): Links to external sources (not directly used for recommendations)
- [`ratings_small.csv`](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset): User ratings (for collaborative filtering and LightFM)

> **Kaggle Dataset Source:** [The Movies Dataset by Rounak Banik](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset)

## Models Used
- **Collaborative Filtering:**
  - Custom neural network (`RecSysModel`) implemented in `recommender.py` (PyTorch/Tez)
  - Learns user and movie embeddings for rating prediction
- **Content-Based Filtering:**
  - TF-IDF vectorization over movie overviews
  - Cosine similarity for finding similar movies
- **Hybrid (LightFM):**
  - [LightFM](https://making.lyst.com/lightfm/docs/home.html) model in `lightfm_recommender.py` (hybrid collaborative + content)
  - Uses WARP loss and both user/movie features

## Installation
1. Clone the repository:
   ```bash
   git clone <repo_url>
   cd Recommender_System-main
   ```
2. Install dependencies (recommended: use a virtual environment):
   ```bash
   pip install -r requirements.txt
   ```

## How to Run

### 1. Streamlit Web App
Launch the interactive UI:
```bash
streamlit run app.py
```

### 2. Train/Evaluate Models
- **Collaborative Filtering:**
  ```bash
  python recommender.py
  ```
  This will train the neural collaborative filtering model and evaluate it.

- **LightFM Hybrid Model:**
  ```bash
  python lightfm_recommender.py
  ```
  This will train and test the LightFM model on your ratings data.

## Directory Structure
```
Recommender_System-main/
├── app.py                  # Streamlit web app
├── recommender.py          # Collaborative/content-based recommenders
├── lightfm_recommender.py  # LightFM hybrid recommender
├── datasets/               # Place all dataset CSVs here
│   ├── movies_metadata.csv
│   ├── keywords.csv
│   ├── links_small.csv
│   └── ratings_small.csv
├── models/                 # Saved models (created after training)
│   ├── recsysmodel.pth
│   └── lightfm_model.pkl
└── README.md
```

## Requirements
- Python 3.7+
- See `requirements.txt` for full list (PyTorch, Tez, LightFM, Streamlit, pandas, scikit-learn, numpy, etc.)

## Notes
- The system now uses `ratings_small.csv` directly for all training and evaluation. No custom split is required by default.
- The hybrid recommender is a placeholder and can be extended for improved results.
- For best results, ensure all dataset files are present in the `datasets/` folder.

## References
- [The Movies Dataset (Kaggle)](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset)
- [LightFM Documentation](https://making.lyst.com/lightfm/docs/home.html)

---

Feel free to contribute or raise issues for improvements!
