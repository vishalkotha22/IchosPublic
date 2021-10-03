from flask import Flask, request, jsonify, render_template

app = Flask(__name__)


@app.route('/')
@app.route('/home')
def home_page():
    return render_template('index.html')

if __name__ == '__main__':
    app.run('0.0.0.0')