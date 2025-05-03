# Bagusdevaa-Sentiments-Analysis-Magangmsib

This repository contains code for sentiment analysis of Indonesian text data, likely related to internship (magang) experiences or MSIB. The project uses data collected from social media Twitter or X, preprocesses it, and applies a sentiment analysis model to classify the text as positive, negative, or neutral. The `process.ipynb` notebook provides the main workflow, while `function.py` contains reusable functions for preprocessing and sentiment prediction.

## Installation

To set up the project, follow these steps:

1.  Clone the repository.
2.  Install the required Python packages using pip:

    ```bash
    pip install -r requirements.txt
    ```
3.  Run the `process.ipynb` notebook to execute the data processing and sentiment analysis pipeline.

## Data

The `data/` directory contains several CSV files with text data from different years, presumably related to "magang" (internship) experiences. Key datasets include:

*   `data/dataset_final.csv`: The final processed dataset with sentiment labels.
*   `data/magang.csv`: Raw tweet data related to "magang" or internship.
*   `data/magang2021.csv`, `data/magang2022.csv`, `data/magang2023.csv`, `data/magang.old.csv`: Various raw tweet datasets from different years.

## Usage

The primary analysis is performed within the `process.ipynb` notebook. This notebook reads the datasets, preprocesses the text data using functions from `function.py`, performs sentiment analysis using a pre-trained model, and generates visualizations.

## Key Files

*   [process.ipynb](https://github.com/Bagusdevaa/Sentiments-Analysis-MagangMSIB/blob/main/process.ipynb): Jupyter Notebook containing the main data processing and analysis workflow.
*   [function.py](https://github.com/Bagusdevaa/Sentiments-Analysis-MagangMSIB/blob/main/function.py): Python script containing functions for text preprocessing and sentiment analysis.
*   [requirements.txt](https://github.com/Bagusdevaa/Sentiments-Analysis-MagangMSIB/blob/main/requirements.txt): List of Python packages required to run the project.
*   [data/dataset_final.csv](https://github.com/Bagusdevaa/Sentiments-Analysis-MagangMSIB/blob/main/data/dataset_final.csv): The final dataset after processing.

## Sentiment Analysis

The sentiment analysis is performed using the `ayameRushia/roberta-base-indonesian-1.5G-sentiment-analysis-smsa` model from Hugging Face Transformers.

## Power BI Dashboard

An interactive dashboard visualizing the sentiment analysis results is available at:

[Interactive Dashboard](https://acmindonesia-my.sharepoint.com/:u:/g/personal/deva999official_member_maribelajar_org/EcDTgx1x4fJCuxTiVmnB-ZMBueqECa8loqsVgzyPju6OVA?e=mjvvJn)
