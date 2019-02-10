"""
    Movie Review Classifier using Keras - Flask Web App

    Flask Web App File

    Author: Sai Hemanth Bheemreddy
"""

from flask import Flask, render_template, request
import model

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    """
    Home Page.

    URL: /
    POST HTTP Method: Renders page along with Keras Model's output
    GET HTTP Method: Renders page without any computation.
    """
    if request.method == 'POST':
        # Retrive review and get rating from model
        review = request.form["review"]        
        rating = model.getRating(review)

        # Open results file to save output for analysis.
        with open("results.csv", "a") as f:
            f.write("{},{}\n".format(review, rating))
        
        # Same IP address and browser information of user
        with open("usage.csv", "a") as f:
            f.write("{}, {}\n".format(request.user_agent.string, request.remote_addr))
        
        # Default message that will overwritten if no error occurs
        result = ["Unexpected Error occured.", "You may have entered a lot of unkonwn words", ""]
        if rating:
            result = ["Review: {}".format(review[:50]),
                        "\nUser has given a {} Review".format('Good' if rating >= 0.5 else 'Bad'), 
                        "Goodness: {:.3f} %".format(rating*100)]

        # Rendering page with result
        return render_template('index.html', result=result)
    else :
        # Rendering page without result
        return render_template('index.html')