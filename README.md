# Student Score prediction System

## 📑 Index

- [Key Highlights](#-key-highlights)
- [Project Overview](#-project-overview)
- [Dataset Description](#-dataset-description)
- [Exploratory Data Analysis (EDA)](#-exploratory-data-analysis-eda)
- [Project Structure](#-project-structure)
- [Explanation of Important Directories & Files](#-explanation-of-important-directories--files)
- [Machine Learning Pipeline Architecture](#-machine-learning-pipeline-architecture)
- [Technology Stack](#️-technology-stack)
- [Local Development Setup & Running the Pipelines](#-local-development-setup--running-the-pipelines)
- [CI/CD & Cloud Deployment Pipeline (Jenkins + Docker + GCP)](#-cicd--cloud-deployment-pipeline-jenkins--docker--gcp)
- [Model Performance & Application Demo](#-model-performance--application-demo)

---

## 🔥 Key Highlights

- Implementation of a complete **End-to-End Machine Learning Pipeline**
- Separate pipelines for:
  - Model Training
  - Model Prediction (Inference)
- Integration of **CI/CD pipeline using Jenkins**
- Full project containerization using Docker
- Deployment-ready Docker images pushed to Google Container Registry (GCR)
- Modular and scalable project architecture

This project demonstrates how a machine learning solution can move from experimentation to a cloud-deployable, production-grade system.

---

## 🎯 Project Overview

The objective of this project is to analyze student demographic and academic attributes — including:

- Gender
- Race/Ethnicity
- Parental Level of Education
- Lunch Type
- Test Preparation Course
- Reading Score
- Writing Score

and predict the **Mathematics score** using supervised machine learning techniques.

Although this implementation focuses on predicting the _math score_ as the target variable, the architecture is flexible and can be adapted to predict either reading or writing scores with minimal modifications.

This project is designed not just as a machine learning model, but as a **production-ready ML system** implementing modern MLOps practices.

---

## 📊 Dataset Description

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>gender</th>
      <th>race_ethnicity</th>
      <th>parental_level_of_education</th>
      <th>lunch</th>
      <th>test_preparation_course</th>
      <th>math_score</th>
      <th>reading_score</th>
      <th>writing_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>female</td>
      <td>group B</td>
      <td>bachelor's degree</td>
      <td>standard</td>
      <td>none</td>
      <td>72</td>
      <td>72</td>
      <td>74</td>
    </tr>
    <tr>
      <th>1</th>
      <td>female</td>
      <td>group C</td>
      <td>some college</td>
      <td>standard</td>
      <td>completed</td>
      <td>69</td>
      <td>90</td>
      <td>88</td>
    </tr>
    <tr>
      <th>2</th>
      <td>female</td>
      <td>group B</td>
      <td>master's degree</td>
      <td>standard</td>
      <td>none</td>
      <td>90</td>
      <td>95</td>
      <td>93</td>
    </tr>
    <tr>
      <th>3</th>
      <td>male</td>
      <td>group A</td>
      <td>associate's degree</td>
      <td>free/reduced</td>
      <td>none</td>
      <td>47</td>
      <td>57</td>
      <td>44</td>
    </tr>
    <tr>
      <th>4</th>
      <td>male</td>
      <td>group C</td>
      <td>some college</td>
      <td>standard</td>
      <td>none</td>
      <td>76</td>
      <td>78</td>
      <td>75</td>
    </tr>
  </tbody>
</table>
</div>

The dataset contains both categorical and numerical features describing student demographics and academic performance.

### 🟢 Categorical Features

These features represent categories or labels:

- gender :— Male or Female
- race_ethnicity :— Group A, B, C, D, or E
- parental_level_of_education :—
  Some high school, High school, Some college, Associate’s degree, Bachelor’s degree, Master’s degree
- lunch :— Standard or Free/Reduced
- test_preparation_course :— Completed or None

These variables describe the background and preparation level of each student.

### 🔵 Numerical Features

These features represent measurable academic scores:

- math_score — Range: 0 to 100 (Target Variable)
- reading_score — Range: 0 to 100
- writing_score — Range: 0 to 100

All numerical features represent exam scores scored out of 100.

---

## 🔎 Exploratory Data Analysis (EDA)

Exploratory Data Analysis (EDA) is a crucial step in building a production-scale machine learning project. It helps us understand the dataset, handle missing or duplicate values, and prepare data for modeling.

### 1️⃣ Basic Dataset Overview

- Check dataset statistics using `df.describe()`:
  - Provides count, mean, std, min, 25%, 50%, 75%, and max for numerical columns.

- Check unique values for categorical variables:

```python
print("Categories in 'gender' variable:     ", df['gender'].unique())
print("Categories in 'race_ethnicity' variable:  ", df['race_ethnicity'].unique())
print("Categories in 'parental_level_of_education' variable:", df['parental_level_of_education'].unique())
print("Categories in 'lunch' variable:     ", df['lunch'].unique())
print("Categories in 'test_preparation_course' variable:     ", df['test_preparation_course'].unique())
```

### 2️⃣ Handling Missing and Categorical Data

- Ensure categorical variables are consistent and impute missing values using SimpleImputer.
- Use column transformation or pipeline to prepare categorical and numerical columns.
- Calculate total and average scores per student for further analysis.

### 3️⃣ Univariate Analysis

- Analyze distributions of numerical features (math_score, reading_score, writing_score).
- Count occurrences of categories in categorical features (gender, race_ethnicity, etc.).
- Visualize using histograms, countplots, and boxplots for insights.

### 4️⃣ Bivariate Analysis

- Examine relationships between categorical variables and the target (math_score):
  - Example: gender vs average_score
  - parental_level_of_education vs average_score
  - lunch vs average_score
- Check correlation between numerical features and target.

### 5️⃣ Multivariate Analysis

- Explore interactions between multiple features and performance.
- Identify patterns, dependencies, and potential predictors for the model.

### 6️⃣ Conclusions

- Student performance is influenced by lunch type, race, and parental education level.
- Females generally have higher scores and pass percentages.
- Test preparation course has less impact overall, but completing it is still beneficial.
- These insights guide feature selection and model preparation.

For more detailed analysis with visualizations, refer to the notebook:
👉 [View EDA Notebook](notebook/EDA_STUDENT_PERFORMANCE.ipynb)

---

## 📁 Project Structure

```
mlproject/
│
├── 🐳 Dockerfile
├── 🔁 Jenkinsfile
├── 📄 README.md
├── 🚀 app.py
├── 📦 requirements.txt
├── ⚙️ setup.py
│
├── 📁 artifacts/
│   ├── 📁 models/
│   │   └── 📄 model.pkl
│   │
│   ├── 📁 processed/
│   │   ├── 📄 processed_train.csv
│   │   └── 📄 processed_test.csv
│   │
│   ├── 📁 raw/
│   │   ├── 📄 stud.csv
│   │   ├── 📄 train.csv
│   │   └── 📄 test.csv
│   │
│   └── 📄 preprocessor.pkl
│
├── 📁 config/
│   ├── 📄 __init__.py
│   └── 📄 paths_config.py
│
├── 📁 custom_jenkins/
│   └── 🐳 Dockerfile
│
├── 📁 logs/
│   └── 📄 log_2026-02-16.log
│
├── 📁 notebook/
│   ├── 📓 EDA_STUDENT_PERFORMANCE.ipynb
│   ├── 📓 MODEL_TRAINING.ipynb
│   ├── 📁 data/
│   │   └── 📄 stud.csv
│   └── 📁 catboost_info/
│
├── 📁 src/
│   ├── 📁 components/
│   │   ├── 📄 __init__.py
│   │   ├── 📄 data_ingestion.py
│   │   ├── 📄 data_transformation.py
│   │   └── 📄 model_trainer.py
│   │
│   ├── 📁 pipeline/
│   │   ├── 📄 __init__.py
│   │   ├── 📄 train_pipeline.py
│   │   └── 📄 predict_pipeline.py
│   │
│   ├── 📄 custom_exception.py
│   ├── 📄 logger.py
│   └── 📄 utils.py
│
├── 📁 static/
│   └── 🎨 style.css
│
├── 📁 templates/
│   └── 🌐 index.html
│
└── 📁 venv/
```

---

## 📌 Explanation of Important Directories & Files

### 🐳 Dockerfile

Defines the container configuration for the application to ensure consistent development and deployment environments.

### 🔁 Jenkinsfile

Defines CI/CD pipeline stages including build, test, Docker image creation, and deployment to cloud.

### 🚀 app.py

Main entry point of the Flask application. Handles user input and triggers the prediction pipeline.

### 📦 artifacts/

Stores all generated outputs from the ML lifecycle.

- `models/` → Contains the trained model (`model.pkl`)
- `processed/` → Processed train and test datasets
- `raw/` → Original dataset files
- `preprocessor.pkl` → Saved preprocessing pipeline object

### ⚙️ config/

Contains configuration files.

- `paths_config.py` → Manages file paths and directory structure across the project.

### 🔧 custom_jenkins/

Contains a dedicated Dockerfile used specifically for Jenkins CI/CD setup.

### 📝 logs/

Stores application logs for debugging and monitoring.

### 📊 notebook/

Contains experimentation and research work.

- `EDA_STUDENT_PERFORMANCE.ipynb` → Exploratory Data Analysis
- `MODEL_TRAINING.ipynb` → Model experimentation
- `data/` → Dataset used for analysis
- `catboost_info/` → CatBoost training logs and metadata

### 🧠 src/

Core source code of the machine learning system.

#### 📁 components/

Contains modular ML components:

- `data_ingestion.py` → Loads and splits raw data
- `data_transformation.py` → Handles preprocessing and feature engineering
- `model_trainer.py` → Trains and evaluates models

#### 📁 pipeline/

Contains pipeline orchestration logic:

- `train_pipeline.py` → Executes full training workflow
- `predict_pipeline.py` → Handles prediction logic for new inputs

- `logger.py` → Custom logging implementation
- `custom_exception.py` → Centralized exception handling
- `utils.py` → Helper functions used across modules

### 🌐 templates/

Contains frontend HTML files.

- `index.html` → User interface for input and prediction display

### 🎨 static/

Contains frontend styling files.

- `style.css` → CSS styling for the web interface

### 🧪 venv/

Local Python virtual environment (not required in production and typically excluded from Git).

---

## 🧠 Machine Learning Pipeline Architecture

> **Note:** Every stage of the pipeline includes logging. If any error occurs, logs can be found in the `logs/` directory. Log files are organized by date for easier debugging and monitoring.

---

### 📥 Data Ingestion

The **Data Ingestion** stage is responsible for loading the dataset using the **Pandas** library and converting it into a DataFrame.

Key steps:

- Reads the dataset from CSV files.
- Splits the dataset into **training and testing sets** using an **80/20 split**.
- Saves the datasets separately inside the `artifacts/raw/` directory.

This stage ensures that raw data is properly organized before moving to the preprocessing stage.

---

### 🔄 Data Transformation

The **Data Transformation** stage prepares the dataset for model training by handling missing values and converting categorical features into numerical representations.

#### Categorical Features

The following features are treated as categorical variables:

- gender
- race_ethnicity
- parental_level_of_education
- lunch
- test_preparation_course

Processing steps:

- **SimpleImputer** → Fills missing categorical values with the most frequent value.
- **OneHotEncoder** → Converts categorical values into numerical vectors.

Example:  
`parental_level_of_education` may contain values like _bachelor's degree_, _master's degree_, _associate degree_, etc. One-hot encoding converts these into binary columns.

> **Note:** Standard scaling is not applied to categorical features because one-hot encoded values are already binary (0 or 1). Applying scaling could distort these values.

---

#### Numerical Features

The following features are treated as numerical variables:

- writing_score
- reading_score

Processing steps:

- **SimpleImputer** → Replaces missing values with the median value.
- **StandardScaler** → Scales values to ensure consistent feature ranges.

---

#### Preprocessing Object

The complete preprocessing pipeline is saved as a **pickle file**:

```
artifacts/preprocessor.pkl
```

This allows the same transformations to be reused during:

- Model training
- Prediction pipeline
- Production deployment

---

### 🤖 Model Training

In this stage, multiple machine learning models are trained using the transformed dataset.

Models used:

- Random Forest Regressor
- Decision Tree Regressor
- Gradient Boosting Regressor
- Linear Regression
- XGBoost Regressor
- AdaBoost Regressor

Each model is trained using the processed dataset produced by the data transformation pipeline.

---

### ⚙️ Hyperparameter Tuning

To improve model performance, selected models are fine-tuned using hyperparameter tuning.

Example parameters explored:

**Random Forest**

- n_estimators

**Gradient Boosting**

- learning_rate
- subsample
- n_estimators

**XGBoost**

- learning_rate
- n_estimators

**AdaBoost**

- learning_rate
- n_estimators

Some models retain default parameters where tuning was not necessary.

---

### 📊 Model Evaluation

All trained models are evaluated using regression metrics.  
The model that achieves the **best performance score** is selected as the final model.

---

### 💾 Model Saving

The best-performing model is saved as:

```
artifacts/models/model.pkl
```

This saved model is later used by the **prediction pipeline** and the **Flask web application**.

---

## ⚙️ Technology Stack

- **Programming Language:** Python
- **Machine Learning:** Scikit-learn, XGBoost
- **Backend Framework:** Flask
- **Frontend:** HTML, CSS
- **API Architecture:** REST API
- **Containerization:** Docker
- **CI/CD:** Jenkins
- **Cloud Platform:** Google Cloud
- **Container Registry:** Google Container Registry (GCR)

---

## 💻 Local Development Setup & Running the Pipelines

Follow the steps below to run the project locally.

---

### 1️⃣ Create a Virtual Environment

First create a Python virtual environment:

```bash
python3 -m venv venv
```

Activate the environment:

```bash
source venv/bin/activate
```

---

### 2️⃣ Verify Project Structure

Before running the application, ensure that all required files and directories are present.  
Refer to the **Project Structure** section in this README for the expected layout.

---

### 3️⃣ Run the Training Pipeline

The **training pipeline** must be executed before running the application.  
This step trains the machine learning model and generates the required pickle files used during prediction.

Run the training pipeline using:

```bash
python -m src.pipeline.train_pipeline
```

After execution, the following files will be created inside the `artifacts/` directory:

- `artifacts/preprocessor.pkl`
- `artifacts/models/model.pkl`

These files are required for the prediction pipeline.

---

### 4️⃣ Run the Application (Prediction Pipeline)

The **prediction pipeline** is integrated into the Flask application (`app.py`), so it does **not need to be executed separately**.

Start the Flask application:

```bash
python app.py
```

Once the server starts, open your browser and navigate to:

```
http://127.0.0.1:8080
```

You can now input student details through the web interface and receive predicted **math scores** generated by the trained model.

---

## 🚀 CI/CD & Cloud Deployment Pipeline (Jenkins + Docker + GCP)

This project implements a **CI/CD pipeline using Jenkins, Docker, and Google Cloud** to automate model training, containerization, and deployment.

The deployment workflow uses the **Docker-in-Docker (DinD)** approach, where Jenkins runs inside a Docker container and builds another container for the ML project.

The automated pipeline follows these stages:

1. **Jenkins Container Setup**
   - Jenkins runs inside a Docker container.
   - Docker CLI is installed inside the Jenkins container to enable Docker-in-Docker builds.

2. **GitHub Integration**
   - Jenkins connects to the GitHub repository using a Personal Access Token.
   - Every commit triggers Jenkins to pull the latest code.

3. **Project Build Pipeline**
   Jenkins executes the following stages:
   - Clone repository
   - Create Python virtual environment
   - Install dependencies
   - Run ML training pipeline
   - Build Docker image
   - Push Docker image to **Google Container Registry (GCR)**
   - Deploy container to **Google Cloud Run**

---

### 🐳 Step 1: Setup Jenkins Container (Docker-in-Docker)

Create a folder:

```
custom_jenkins/
```

Inside it create a **Dockerfile**.

#### Jenkins Dockerfile

```dockerfile
FROM jenkins/jenkins:lts

USER root

RUN apt-get update && \
    apt-get install -y \
    ca-certificates \
    curl \
    gnupg \
    lsb-release && \
    mkdir -p /etc/apt/keyrings && \
    curl -fsSL https://download.docker.com/linux/debian/gpg | \
    gpg --dearmor -o /etc/apt/keyrings/docker.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
    https://download.docker.com/linux/debian \
    $(lsb_release -cs) stable" \
    > /etc/apt/sources.list.d/docker.list && \
    apt-get update && \
    apt-get install -y docker-ce-cli

RUN groupadd -f docker && \
    usermod -aG docker jenkins

USER jenkins
```

---

#### Build Jenkins Image

```bash
docker build -t jenkins-dind .
```

#### Run Jenkins Container

```bash
docker run -d \
--name jenkins-dind \
--privileged \
-p 8080:8080 \
-p 50000:50000 \
-v /var/run/docker.sock:/var/run/docker.sock \
-v jenkins_home:/var/jenkins_home \
jenkins-dind
```

Open Jenkins UI:

```
http://localhost:8080
```

Retrieve the initial admin password:

```bash
docker logs jenkins-dind
```

Install **Suggested Plugins** and create the admin user.

---

### 🔗 Step 2: Connect Jenkins to GitHub

Generate a **GitHub Personal Access Token**.

GitHub → Settings → Developer Settings → Personal Access Tokens → Classic Token

Permissions required:

- `repo`
- `admin:repo_hook`

Add this token to Jenkins:

```
Manage Jenkins
→ Credentials
→ Global
→ Add Credentials
```

Use **Username + Password**:

- Username → GitHub username
- Password → GitHub token

▶️ Video Tutorial

https://youtube.com/YOUR_GITHUB_TOKEN_VIDEO

This tutorial explains how to generate a **GitHub Personal Access Token** and configure it in Jenkins.

---

### 🐳 Step 3: Dockerize the ML Project

Create the main **Dockerfile** for the ML application.

```dockerfile
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY . .

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -e .

ENV PORT=8080
EXPOSE 8080

CMD ["python", "app.py"]
```

---

### ☁️ Step 4: Install Google Cloud CLI in Jenkins

Enter Jenkins container:

```bash
docker exec -u root -it jenkins-dind bash
```

Install Google Cloud CLI:

```bash
apt-get update
apt-get install -y curl gnupg ca-certificates

mkdir -p /usr/share/keyrings

curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg \
 | gpg --dearmor -o /usr/share/keyrings/google-cloud.gpg

echo "deb [signed-by=/usr/share/keyrings/google-cloud.gpg] https://packages.cloud.google.com/apt cloud-sdk main" \
 > /etc/apt/sources.list.d/google-cloud-sdk.list

apt-get update
apt-get install -y google-cloud-cli
```

Verify installation:

```
gcloud --version
```

---

### ☁️ Step 5: Google Cloud Setup

1. Create a **Google Cloud Project**
2. Create a **Service Account**
3. Assign roles:
   - Owner (for development)
   - Cloud Run Admin
   - Storage Admin
4. Download the **service account JSON key**.

▶️ Video Tutorial

https://youtube.com/YOUR_GCP_SETUP_VIDEO

This tutorial shows how to configure Google Cloud for the deployment pipeline.

Add it to Jenkins:

```
Manage Jenkins
→ Credentials
→ Global
→ Add Credential
→ Secret File
```

Upload the JSON key.

▶️ Video Tutorial

https://youtube.com/YOUR_SECRET_FILE_VIDEO

This tutorial explains how to create a **Google Cloud service account JSON key** and add it as a credential in Jenkins.

Enable required APIs:

- Google Container Registry API
- Artifact Registry API
- Cloud Resource Manager API
- Cloud Run API

---

### 🔁 Step 6: Jenkins Pipeline (Jenkinsfile)

The Jenkins pipeline automates the following tasks:

- Clone GitHub repository
- Create Python virtual environment
- Install dependencies
- Run ML training pipeline
- Build Docker image
- Push image to **Google Container Registry**
- Deploy application to **Google Cloud Run**

Example pipeline stages:

```
Clone Repository
↓
Create Virtual Environment
↓
Run Training Pipeline
↓
Build Docker Image
↓
Push Image to GCR
↓
Deploy to Cloud Run
```

Run the pipeline by clicking **Build Now** in Jenkins.

▶️ Video Tutorial

https://youtube.com/YOUR_PIPELINE_VIDEO

This tutorial demonstrates how to create the Jenkins pipeline and run the CI/CD build process.

---

### ☁️ Final Deployment Flow

```
GitHub Repository
        │
        ▼
     Jenkins
 (CI/CD Pipeline)
        │
        ▼
 Docker Image Build
        │
        ▼
 Google Container Registry
        │
        ▼
   Cloud Run Deployment
        │
        ▼
   Public ML Web Application
```

---

### ⚠️ Debugging Tip

When developing the Jenkins pipeline, it is recommended to **test stages incrementally**:

1. First test repository checkout
2. Then test virtual environment setup
3. Then training pipeline
4. Then Docker build
5. Finally GCR push and Cloud Run deployment

This helps identify configuration issues quickly.

---

## 📈 Model Performance & Application Demo

### 📊 Model Performance

Multiple regression models were trained and evaluated using the **R² Score** metric.  
The model with the highest performance was selected as the final model for deployment.

| Model Name              | R² Score |
| ----------------------- | -------- |
| Ridge                   | 0.880593 |
| Linear Regression       | 0.880433 |
| CatBoosting Regressor   | 0.851632 |
| Random Forest Regressor | 0.848711 |
| AdaBoost Regressor      | 0.847830 |
| XGBRegressor            | 0.827797 |
| Lasso                   | 0.825320 |
| K-Neighbors Regressor   | 0.783681 |
| Decision Tree           | 0.755587 |

From the evaluation results, **Ridge Regression** achieved the highest **R² score (0.880593)** and was therefore selected as the final model for prediction.

---

### 🎥 Application Demo

The video below demonstrates the working web application where users input student details and receive predicted **math scores**.

▶️ Watch the demo video here:

https://youtube.com/YOUR_APP_DEMO_VIDEO_LINK

This demonstration shows:

- Running the Flask web application
- Entering student details through the UI
- Generating predicted math scores
- End-to-end inference pipeline execution
