from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
app = Flask(__name__)

@app.route('/')
def home():
   return render_template('index.html')

@app.route('/alzheimers')
def alzheimers():
   return render_template('alzheimers.html')

@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
   if request.method == 'POST':
      f = request.files['file']
      f.save(secure_filename(f.filename))
      return "file uploaded succesfully"

@app.route('/sli')
def sli():
   return render_template('sli.html')

@app.route('/respiratory')
def respiratory():
   return render_template('respiratory.html')

@app.route('/generic')
def generic():
   return render_template('generic.html')

if __name__ == '__main__':
   app.run()