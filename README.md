# TruthDrift: AI Hallucination Detection System
This project is an ML-based verification system which detects hallucinations in AI-generated text. This is via verifying if text fed in the model are supported by accurate text derived from the public FEVER dataset. Features of the ML model to classify claims as hallucinated or correct include TF-IDF, embedding similarity, and negation mismatch.

## Overview
TruthDrift is a machine learning system designed to detect hallucinations in AI-generated text. The system classifies sentences as either factual or hallucinated using multiple feature types.

The model combines:
- TF-IDF (lexical features)
- Jaccard similarity (word overlap)
- Cosine similarity (centroid similarity)
- GloVe embeddings (semantic similarity)

A Streamlit UI allows users to input text and receive predictions with confidence scores.

## Code Structure

TruthDrift/
- TruthDriftModel.py - Training pipeline
- predict.py - Inference pipeline (used by UI)
- app.py - Streamlit user interface
- graphsAnalysis.py - Generates graphs (accuracy, F1, confusion matrix)
- requirements.txt - Dependencies
- README.md - Project documentation and setup details

## Dependencies

Install all required packages:
pip install -r requirements.txt

Specified ibraries used (And listed in requirements.txt):
- numpy
- scikit-learn
- scipy
- matplotlib
- streamlit

## How to Run the Project

Step 0 - Download necessary packages and features (located in requirements.txt)

Step 1 — Enter the following in your terminal: 
python TruthDriftModel.py

This line of code will:
- Train the model
- Generate .pkl files (these are the model and preprocessing components)

Step 2 — Run the UI to scan AI generated text by entering the following in your terminal: 
streamlit run app.py

Then enter the following link in your browser: http://localhost:8501

Step 3 — Generate Graphs
python graphsAnalysis.py

This will generate:
- accuracy_f1_graph.png (Provides accuracy and f1 scores for 4 feature combinations. )
- confusion_matrix.png (Summarizing the full model’s performance via showing how many predictions were correct and incorrect across each class)

## Dataset and Model Requirements

FEVER Dataset: Already within the Github Datasets folder. 

GloVe Embeddings:
Download:
https://github.com/stanfordnlp/glove -> Wikipedia 2014 + Gigaword 5 (6B tokens, 400K vocab, uncased, 300d vectors, 822 MB download): glove.6B.zip

Extract from opened zip file:
glove.6B.50d.txt

Place that dataset dataset (Ensure title is exactly as above) in:
Datasets/

## *Note on Dataset Availability

GloVe embeddings specifically NOT included in this repository because:
- GitHub has a 100MB file size limit
- The GloVe file specifically was too large to store in the repository. 

Anybody interested in using TruthDrift must manually download the file and place it in the Datasets folder using the instructions above.

## Code Authorship and Attribution

Code Fully Written by Author (Sharvari Deshpande):
- Feature engineering (TF-IDF, Jaccard similarity, cosine similarity, GloVe embeddings)
- Logistic regression training pipeline
- Feature combination and scaling
- Inference pipeline (predict.py)
- Graph generation (graphsAnalysis.py)
- UI for user to add AI generated text(app.py)

ML Concepts for Project (Not Code) have been Derived From :
- TF-IDF and cosine similarity (standard NLP techniques)
- Logistic regression (scikit-learn documentation)
- GloVe embeddings (Stanford NLP resources)

There was no code that was copied from external repositories.

## Reproducibility

To reproduce results:
1. Download datasets and embeddings
2. Run training script
3. Run graph analysis
4. Run UI

All outputs will be generated locally.

## Limitations

- The model does not use external knowledge retrieval
- Performance is limited by feature-based methods
- Semantic embeddings provide limited improvement compared to cost

## Future Work

- Integrate retrieval-based verification
- Use transformer-based embeddings (e.g., BERT)
- Improve confidence calibration