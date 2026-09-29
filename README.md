# AspectSense

## Aspect-Based Sentiment Analysis using DistilBERT

AspectSense is an Aspect-Based Sentiment Analysis (ABSA) system that predicts the sentiment associated with a specific aspect mentioned in a review.

## Problem

A single review can contain different sentiments for different aspects.

Example:

> "The camera is excellent but the battery life is poor."

- Aspect: camera → Positive
- Aspect: battery life → Negative

AspectSense focuses on the specified aspect instead of predicting the sentiment of the entire review.

## Technology Used

- Python
- PyTorch
- Hugging Face Transformers
- DistilBERT
- Streamlit
- Pandas
- Scikit-learn
- Plotly

## Model

The system uses a pretrained DistilBERT transformer with aspect-conditioned input.

The aspect term and review text are provided to the model as a pair.

The model predicts one of three sentiment classes:

- Negative
- Neutral
- Positive

## Features

- Single review sentiment analysis
- Aspect-specific sentiment prediction
- Confidence scores
- Admin dashboard
- Sentiment distribution charts
- Batch prediction for test data
- Submission CSV generation

## Project Structure

```text
AspectSense/
├── app.py
├── train_model.py
├── requirements.txt
└── README.md
