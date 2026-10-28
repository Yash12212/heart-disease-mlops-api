# Heart Disease Prediction API

An end-to-end machine learning project: a model is trained in **Google Colab**
on a **Kaggle** dataset, saved to disk, and served as a **FastAPI** REST API that
is packaged with **Docker** and validated by a **GitHub Actions** CI pipeline.

## Dataset

- **Source:** [Heart Disease Dataset on Kaggle](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)
- **Author:** johnsmith88
- **File:** `heart.csv` (1025 rows, 13 features, 1 target)
- **Target:** `target` — `0` = no heart disease, `1` = heart disease

The dataset is downloaded from Kaggle and stored in the `data/` folder.

## Project workflow

1. **Get the data** — download `heart.csv` from the Kaggle dataset page above.
2. **Train the model in Colab** — open `notebook/Heart_Disease_Model_Training.ipynb`
   in Google Colab, upload `heart.csv` to the Colab session, and run all the
   cells. The notebook:
   - explores the data,
   - splits it into train / test sets,
   - trains a scikit-learn `Pipeline` (impute → scale → RandomForest),
   - evaluates it on the test set,
   - saves the model to `model/model.joblib` and the results to `model/metrics.json`,
   - reloads the saved model and makes a test prediction.
3. **Serve the model** — the FastAPI app in `app/` loads `model.joblib` and
   exposes prediction endpoints.
4. **Package with Docker** — the `Dockerfile` builds a container that runs the API.
5. **Automate with CI** — `.github/workflows/ci.yml` runs the tests and builds
   the Docker image on every push and pull request.

## Project structure

```
.
├── notebook/
│   ├── Heart_Disease_Model_Training.ipynb          # training notebook (run in Colab)
│   └── Heart_Disease_Model_Training_output.ipynb   # executed notebook with results
├── model/
│   ├── model.joblib                                # trained model
│   └── metrics.json                                # model parameters and accuracy
├── app/
│   ├── main.py                                     # FastAPI application
│   └── schemas.py                                  # request / response models
├── tests/
│   └── test_api.py                                 # automated API tests
├── data/
│   └── heart.csv                                   # dataset downloaded from Kaggle
├── Dockerfile
├── requirements.txt
└── .github/workflows/ci.yml
```

## Model

- **Algorithm:** Random Forest classifier (200 trees)
- **Preprocessing:** median imputation + standard scaling, bundled in a pipeline
- **Test accuracy:** see `model/metrics.json`

## Running the API locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Then open http://localhost:8000/docs for the interactive API documentation.

## Running with Docker

```bash
docker build -t heart-disease-ml-api .
docker run -p 8000:8000 heart-disease-ml-api
```

## API endpoints

| Method | Path       | Description                          |
|--------|------------|--------------------------------------|
| GET    | `/`        | Welcome message                      |
| GET    | `/health`  | Health check                         |
| POST   | `/predict` | Predict heart disease for one patient|

### Example request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age":52,"sex":1,"cp":0,"trestbps":125,"chol":212,"fbs":0,"restecg":1,"thalach":168,"exang":0,"oldpeak":1.0,"slope":2,"ca":2,"thal":3}'
```

### Example response

```json
{"prediction": 0, "label": "No disease", "probability": 0.985}
```

## Tests

```bash
python -m pytest -v
```

## CI

Every push or pull request to `main` triggers the GitHub Actions workflow, which
installs the dependencies, runs the tests with `pytest`, and builds the Docker
image to confirm the application packages correctly.
