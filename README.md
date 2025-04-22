# World Happiness Report – Clustering & Regression Project

This project analyzes the World Happiness Report 2024 dataset to uncover patterns in country-level quality of life indicators.  
It combines unsupervised learning (GMM clustering) with regression analysis to predict life satisfaction scores (`ladder_score`).

## Key Features
- Data cleaning and preprocessing
- GMM clustering of countries based on 6 socio-economic indicators
- Analysis of group differences (mean vs global average, standard deviation)
- Visualizations of group patterns
- Linear regression to predict `ladder_score`
- Evaluation with MAE and R² metrics

## Technologies
- Python
- pandas, scikit-learn, matplotlib, seaborn

## Running the Project
1. Clone the repo  
2. Install requirements: `pip install -r requirements.txt`  
3. Run `main.py`

## Project Structure
- `get_data.py` – Loads and cleans the WHR dataset, and renames columns for consistency.
- `group_analysis.py` – Performs clustering (GMM), calculates group-level means, standard deviations,
   and generates visualizations.
- `regression_model.py` – Trains a linear regression model to predict the happiness score (`ladder_score`) and 
   evaluates it using MAE and R².
- `main.py` – Main script that orchestrates all steps: loading data, clustering, analysis, prediction, and output
   generation.
- `README.md` – This file 😊
- `requirements.txt` – Lists the Python libraries required to run the project.


## Author
Marina Kurland 💛
