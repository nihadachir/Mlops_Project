# Wine Quality Prediction with MLflow

This project is an **end-to-end machine learning pipeline** for predicting **wine quality** based on physicochemical features. It includes **data preprocessing, model training, evaluation, and MLflow experiment tracking**, along with a local web app (`app.py`) for making predictions.

---

## Project Workflows

1. Update `config.yaml`  
2. Update `schema.yaml`  
3. Update `params.yaml`  
4. Update the entity  
5. Update the configuration manager in `src/config`  
6. Update components in `src/components`  
7. Update the pipeline in `src/pipeline`  
8. Update `main.py`  
9. Update `app.py`  

---

## How to Run

### Step 1: Clone the repository

```bash
git clone https://github.com/nihadachir/Mlops_Project.git
cd Mlops_Project
Step 2: Create and activate a conda environment
bash
Copier le code
conda create -n mlproj python=3.8 -y
conda activate mlproj
Step 3: Install dependencies
bash
Copier le code
pip install -r requirements.txt
Step 4: Configure MLflow (Optional)
You can track experiments locally or using DagsHub.

For DagsHub:

bash
Copier le code
export MLFLOW_TRACKING_URI=https://dagshub.com/<username>/<repo>.mlflow
export MLFLOW_TRACKING_USERNAME=<username>
export MLFLOW_TRACKING_PASSWORD=<personal-access-token>
For Local MLflow:

bash
Copier le code
mlflow ui --backend-store-uri ./mlruns --default-artifact-root ./mlruns --host 0.0.0.0 --port 5000
Visit: http://127.0.0.1:5000 to access the MLflow UI.

Step 5: Run the Web App
bash
Copier le code
python app.py
Open your browser at http://127.0.0.1:5000 (or the port specified in app.py).

Input wine features and get the predicted wine quality score.

Optional: Run the Pipeline
bash
Copier le code
python main.py
Executes the full pipeline end-to-end: data preprocessing → training → evaluation → MLflow logging.

MLflow
Track experiments: Log parameters, metrics, and trained models.

Visualize results: Use MLflow UI locally or via DagsHub.

Official MLflow Documentation