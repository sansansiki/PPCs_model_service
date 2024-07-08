python = 3.9.7

## model_example.ipynb :

Given the test data, the model infers the probability of the resulting outcome.

## app

deployed model by sending an HTTP request to the Flask application's URL. (eg. 127.0.0.1:5000/ppcs/predict)

> PORT: 5000 -> model_service/app/config.py

**run:**

> cd app
>
> python run app.py
