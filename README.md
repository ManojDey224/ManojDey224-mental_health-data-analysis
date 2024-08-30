# Mental Health Status Prediction from Text

This project uses machine learning to predict an individual's mental health status based on their written statements. The goal is to explore how natural language processing (NLP) can be applied to help identify potential mental health concerns.

## Dataset

The dataset used for this project is a combined dataset from various sources related to mental health. It contains the following columns:

- **original_statement:** The original statement provided by the individual.
- **status:** The labeled mental health status associated with the statement.

## Project Workflow

1. **Data Loading and Preprocessing:**
   - Load the dataset from a CSV file.
   - Clean the data by removing missing values, converting statements to lowercase, and removing irrelevant patterns (URLs, handles, etc.).
2. **Feature Engineering:**
   - Tokenize the statements using NLTK's word_tokenize.
   - Stem the tokens using NLTK's PorterStemmer.
   - Extract numerical features such as the number of characters and sentences in each statement.
3. **Text Vectorization:**
   - Use TF-IDF (Term Frequency-Inverse Document Frequency) to convert the stemmed tokens into numerical vectors. This process captures the importance of words in a statement relative to the entire corpus.
   - Combine the TF-IDF vectors with the extracted numerical features to create a comprehensive feature representation.
4. **Handling Class Imbalance:**
   - Apply Random Over-Sampling using imblearn's RandomOverSampler to address class imbalance in the dataset. This technique duplicates samples from the minority class to balance the class distribution.
5. **Model Training and Evaluation:**
   - Train different classifiers on the resampled training data:
     - Bernoulli Naive Bayes
     - Decision Tree
     - Logistic Regression
     - XGBoost 
   - Evaluate the performance of each model on the test data using metrics such as accuracy, precision, recall, and F1-score. Generate classification reports and confusion matrices to visualize model performance for each class. 
6. **Visualization:**
   - Create word clouds for each mental health status category to visualize common terms associated with different statuses.
   - Visualize a bar chart to compare the accuracy scores of each trained classifier, aiding in selecting the best model.

## Code and Requirements

The project is implemented in a Python script (`mental_health.py`). You can find the script in the repository along with a copy of the original Google Colab notebook file used (`mental_health.ipynb`) The code relies on several libraries including:

- pandas
- numpy
- matplotlib
- seaborn
- nltk
- imblearn
- scikit-learn
- xgboost-gpu

You can install these dependencies using: `pip install -r requirements.txt`

## Running the Project

1. Clone this repository: `git clone <repository-url>`
2. Navigate to the project directory: `cd mental_health_prediction`
3. Install the necessary dependencies: `pip install -r requirements.txt`
4. Execute the script: `python mental_health.py`

The script will perform all the analysis steps, generate the visualizations, and print the results to the console.

## Results and Discussion 

The results (accuracy, classification report, confusion matrices) for each of the trained classifiers will be displayed in the console. You can then compare the models to see which one performs best. Additional discussion and potential explanations for these results can be found within comments in the code file itself. 

## Further Improvements 

- **Experiment with More Advanced NLP Techniques:**  Explore different text preprocessing steps such as lemmatization, POS tagging, and named entity recognition to refine the features extracted from text.
- **Hyperparameter Tuning:** Use GridSearchCV to fine-tune the hyperparameters of each model, seeking to find the most optimal settings and potentially improve performance. 
- **Deploy as an API or Application:** To make this more accessible, consider wrapping the trained model (the best performer) into an API or develop a user interface to allow others to input statements and receive mental health status predictions. 

## Contribution 

Feel free to fork this repository and contribute!  Some ideas for improvement include:
- Expanding the dataset to include more diverse and inclusive examples. 
- Testing the model with different types of text input. 
- Creating interactive visualizations to display the analysis results.

## License 

This project is licensed under the MIT License. You are welcome to modify and redistribute this code for both personal and commercial use.  See `LICENSE` for details.
