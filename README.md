### Prerequisites
You must have Scikit Learn, Pandas (for Machine Leraning Model) and Flask (for API) installed.

### Project Structure
1. model.py - This contains code for our Machine Learning model to perform stock predictions based on the past training data in 'BTC-USD.csv', etc... files from yahoo finance for quick prototyping.

2. app.py - This contains Flask APIs that receives employee details through GUI or API calls, computes the precited value based on our model and returns it.

3. templates - This folder contains the HTML template to allow user to enter employee detail and displays the predicted employee salary.

### Running the project
1. Ensure that you are in the project home directory. Create the machine learning model by running below command -
```
python model.py, eth_model, etc...
```
This would create a serialized version of our model into a file model.pkl

2. Run app.py using below command to start Flask API
```
python app.py
```
By default, flask will run on port 5000.

3. Navigate to URL http://localhost:5000

CHOOSE STOCK YOU WOULD LIKE TO PREDICT :
![alt text](https://i.imgur.com/ZI1FKwQ.png)

Next, WE PAY FOR THE STOCK WE WOULD LIKE TO PREDICT:
![alt text](https://i.imgur.com/rX93R1n.png)

AFTERWARDS, THE STOCK PREDICTION WILL BE OUTPUTTED AFTER PAYMENT!
![alt text](https://i.imgur.com/PrKA4zP.png)

THE FINAL OUTPUT IS THE HEROKU HOST : https://stock-prediction-basic.herokuapp.com/
