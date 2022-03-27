
from flask import Flask, render_template,request
import  numpy  as np
from joblib import load
import os
from statistics import mode
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///predictions.sqlite3'  # step 2
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

#  database class
db = SQLAlchemy(app) # step 3

# step 4 create a table using python class
class PredHistory(db.Model):
	id = db.Column(db.Integer, primary_key=True)
	age = db.Column(db.Integer)
	salary = db.Column(db.Float)
	prediction = db.Column(db.String)

	def __str__(self):
		return f"{self.id}, {self.age}, {self.salary}, {self.prediction}"


#  end of database class
# step 5 is on terminal

def load_clf_model():
  print(os.listdir())
  filepath='Machine-Learning/app_1/clf_ap.pkl'
  return load(filepath)

def predict(age=0,salary=0):
  userinp=np.array([[age,salary]])
  model_dict=load_clf_model()

  x=model_dict.get('scaler').transform(userinp)
  p=model_dict.get('classifier').predict(x)
  if p[0]==0:
    return "will not purchased"
  else:
    return 'will make purchase'  
  

@app.route('/',methods=['GET','POST'])
def index():
  if request.method=='POST':
      form=request.form
      age=int(form.get('age'))
      salary=float(form.get('salary'))
      userinp=np.array([[age,salary]])
      result=predict(age,salary)
      return render_template('index.html',age=age,salary=salary,result=result)

  return render_template('index.html')

if __name__ == '__main__':
  app.run(debug=True)
 