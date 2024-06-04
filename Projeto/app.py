from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)


with open('vgame_sales.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        if not file:
            return render_template('index.html', prediction="No file uploaded. Please upload a CSV file.")
        
       
        df = pd.read_csv(file)
        
      
        required_columns = ['Platform', 'Year', 'Genre']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return render_template('index.html', prediction=f"Missing columns: {', '.join(missing_columns)}")
        
        
        predictions = model.predict(df)
        
       
        result_df = df.copy()
        result_df['Predicted_Sales'] = predictions
        
        
        result_html = result_df.to_html(index=False)
        
        return render_template('index.html', prediction=result_html)

if __name__ == '__main__':
    app.run(debug=True)
