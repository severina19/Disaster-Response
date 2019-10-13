import sys
from sqlalchemy import create_engine
import pandas as pd
import nltk
nltk.download(['punkt', 'wordnet', 'stopwords'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
from nltk.corpus import stopwords
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, make_scorer
import pickle
stopWords = stopwords.words('english')


def load_data(database_filepath):
    engine = create_engine(f'sqlite:///{database_filepath}')
    conn = engine.connect()
    df = pd.read_sql_table('DisasterResponse', con = conn)
    
    X = df['message'].values
    drop_columns = ['id','message', 'original', 'genre']
    category_names = df.drop(columns=drop_columns).columns
    Y = df.drop(columns=drop_columns).values
    
    return X, Y, category_names


def tokenize (text):
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    tokens = word_tokenize(text)
    tokens = [k for k in tokens if k not in stopWords]
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

def multiclass_f1_score(y_true, y_pred):
    """Calculate mean F1 score for all of the output classifiers
    """
    f1_list = []
    for i in range(np.shape(y_pred)[1]):
        f1 = f1_score(np.array(y_true)[:, i], y_pred[:, i], average='micro')
        f1_list.append(f1)
        
    score = np.mean(f1_list)
    return score

def build_model(gridsearch=False):
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    if not gridsearch:
        return pipeline

    parameters = {'vect__min_df': [1, 5],
              'clf__estimator__n_estimators':[10, 25], 
              'clf__estimator__min_samples_split':[2, 10]}
    scorer = make_scorer(multiclass_f1_score)
    cv = GridSearchCV(pipeline, param_grid=parameters,scoring = scorer, verbose=10, n_jobs=10)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    # Calculate evaluation metrics for each set of labels
    Y_pred = model.predict(X_test)
    for i in range(len(category_names)):
        precision = precision_score(Y_test[:, i], Y_pred[:, i], average='micro')
        recall = recall_score(Y_test[:, i], Y_pred[:, i], average='micro')
        f1 = f1_score(Y_test[:, i], Y_pred[:, i], average='micro')
        print(f'For category {category_names[i]}, the precision is {precision}, '
              f'recall is {recall}, f1 score is {f1}. ')


def save_model(model, model_filepath):
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
	# gridsearch set to False since it takes a long time to perform
	# But you should activate GridSearch since it increases accuracy
        gridsearch = False
        model = build_model(gridsearch)
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()