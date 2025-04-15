Quora Question Pair Similarity

This project aims to identify duplicate question pairs on the Quora platform by measuring the semantic similarity between question pairs using machine learning and natural language processing (NLP) techniques.

Features
•	Data Preprocessing: Text cleaning and feature extraction for question pairs.
•	Feature Extraction: Utilizes advanced NLP techniques such as TF-IDF, word embeddings (GloVe), and similarity measures.
•	Modeling: Implements various machine learning models such as:
o	BERT (Bidirectional Encoder Representations from Transformers)
o	LSTM (Long Short-Term Memory) networks
o	Logistic Regression
o	Linear SVM
o	XGBoost

•	Evaluation: The models are evaluated using accuracy, precision, recall, F1-score, and log loss.

Dataset
The project uses the Quora Question Pairs Dataset from Kaggle, which consists of question pairs labeled as "duplicate" or "not duplicate."

Technologies Used
•	Python
•	Pandas & NumPy
•	Scikit-learn
•	TensorFlow/Keras (for LSTM)
•	Hugging Face Transformers (for BERT)
•	XGBoost
•	Matplotlib & Seaborn (for visualizations)
