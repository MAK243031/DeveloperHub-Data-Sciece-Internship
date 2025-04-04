# DeveloperHub-Data-Sciece-Internship
Task 4 Sentiment Analysis

# Sentiment Analysis of Movie Reviews

This project demonstrates sentiment analysis of movie reviews using Natural Language Processing (NLP) and machine learning in Python.

## Project Overview

The goal is to build a model that can automatically classify movie reviews as either positive or negative. This is achieved by utilizing NLP techniques to preprocess text data and then training a machine learning model to make predictions.

## Dataset

The project uses the "aclImdb" dataset, a widely used collection of movie reviews for sentiment analysis tasks.

## Methodology

1. **Text Preprocessing:**
   - **Tokenization:** Splitting reviews into individual words.
   - **Stop Word Removal:** Removing common words (e.g., "the," "a," "is") that don't carry much meaning.
   - **Lemmatization:** Reducing words to their base form (e.g., "running" to "run").

2. **Feature Engineering:**
   - **TF-IDF Vectorization:** Converting text into numerical representations using TF-IDF (Term Frequency-Inverse Document Frequency) to capture word importance.

3. **Model Training:**
   - **Logistic Regression:** A machine learning model is trained to classify reviews based on their TF-IDF vectors.

4. **Model Evaluation:**
   - **Classification Report:** The model's performance is evaluated using metrics like precision, recall, and F1-score to measure its accuracy in classifying positive and negative reviews.

5. **Custom Prediction:**
   - A function `predict_sentiment(text)` allows users to input their own text and get a sentiment prediction.

## Requirements

- Python 3.x
- Libraries: nltk, scikit-learn, pandas

## Usage

1. Clone the repository: `git clone <repository_url>`
2. Install the required libraries: `pip install nltk scikit-learn pandas`
3. Run the Jupyter Notebook: `jupyter notebook Sentiment_Analysis.ipynb`

## Results

The model achieves a good accuracy in classifying movie reviews as positive or negative. Refer to the 'Model Evaluation' section in the notebook for detailed results.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

This project is licensed under the MIT License.

## Acknowledgments

- The "aclImdb" dataset creators.
- The developers of the NLP and machine learning libraries used in this project.


