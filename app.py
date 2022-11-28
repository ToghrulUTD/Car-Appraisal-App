from flask import Flask, render_template, request
import numpy as np
import kaleido
from joblib import load
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import uuid

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def hello_world():
    request_type = request.method
    if request_type == 'GET':
        return render_template('index.html', href='static/make_prices.svg')
    else:
        try:
            # get the input text for the required fields
            text = [x.lower().strip() for x in request.form.values()]
            new_values = {'make': [text[0]], 'model':[text[1]], 'type':[text[2]],
                          'year':[float(text[3])], 'mileage':[float(text[4])], 'zip':[float(text[5])]}
            new_values['mile_per_year'] = [new_values['mileage'][0]/(2022- new_values['year'][0])]
            # random string for the pathname
            random_string = uuid.uuid4().hex
            path = "static/" + random_string + ".svg"
            # generate prediction plot
            make_picture(new_values=new_values, output_file=path)
            return render_template('index.html', href=path)
        except:
            return render_template('index.html', href='static/img.png')


# generate prediction and plot
def make_picture(new_values=None, training_data_path='cars_data.csv', model_path='final_model.pkl',
                 output_file='predictions_pic.svg'):
    # inputs
    car_make = new_values['make'][0]
    car_model = new_values['model'][0]
    # load the data
    data = pd.read_csv(training_data_path)
    # create a subset for given model and make
    subset = data.sample(100000)
    subset = subset[(subset.make == car_make) & (subset.model == car_model)]
    # get prediction
    ## 1) convert inputs into suitable df format
    df = pd.DataFrame(new_values)
    features = ['type', 'model', 'make', 'year', 'mileage', 'mile_per_year', 'zip']
    df = df.loc[:, features]
    print(df)
    for column in ['type', 'model', 'make', 'zip']:
        df[column] = df[column].astype('category')
    ## 2) load the model and make prediction
    model = load(model_path)
    pred = model.predict(df)[0]

    # plot the prices as a function of mileage for the given model
    fig = px.scatter(
        subset, x="year", y="price", color="mileage", hover_name='type',
        title=f'Price of {car_make}-{car_model} by mileage and year',
        trendline='ols', trendline_options=dict(log_y=True)  # or trendline = 'lowess'
    )

    fig.add_trace(
        go.Scatter(
            x=df['year'], y=[pred],
            mode="markers + text", name='',
            marker={'color': 'green', 'size': 15},
            text=[f"Estimated price = {str(round(pred / 1000, 1)) + ' k'}"],
            textposition="top left",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df['year'], y=[pred],
            mode="markers", name='',
            marker={'color': 'green', 'size': 15},
            text=[f"Mileage = {round(new_values['mileage'][0] / 1000, 1)}k, body-style = {new_values['type'][0]}"],
            textposition="top left",
        )
    )
    fig.update_layout(showlegend=False)
    fig.write_image(output_file, width=800, engine='kaleido')
    fig.show()
    return fig