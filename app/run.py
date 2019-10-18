import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('DisasterResponse', engine)

# load model
model = joblib.load("../models/disaster_model_best_moclassifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # extract data used for visualization
    genre_request = df[df['request']==1].groupby('genre').count()['message']
    genre_not_req = df[df['request']==0].groupby('genre').count()['message']
    genres = list(genre_request.index)
    
    # Calculate occurance percentage of each message category 
    cat_percentage = df.drop(columns=['id', 'message', 'original', 'genre', 'related']).sum()/len(df)
    cat_percentage = cat_percentage.sort_values(ascending = False)
    categories = list(cat_percentage.index)
     
    # visualization
    graphs = [
        {
            'data': [
                Bar(
                    x=genres,
                    y=genre_request,
                    name = 'Request'
                ),
                
                Bar(
                    x=genres,
                    y=genre_not_req,
                    name = 'Not Request'
                )
            ],

            'layout': {
                'title': 'Number of Messages by Genre ',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                },
                'barmode': 'group'
            }
        },
        {
            'data': [
                Bar(
                    x=categories,
                    y=cat_percentage
                )
            ],

            'layout': {
                'title': 'Percentage of Messages by Category',
                'yaxis': {
                    'title': "Percentage"
                },
                'xaxis': {
                    'title': "Category",
                    'tickangle': -45
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()