import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import warnings
import sys
warnings.filterwarnings('ignore')

def generate_sample_data(n_samples=1000):
    np.random.seed(42)
    
    data = {
        'sqft': np.random.randint(500, 5000, n_samples),
        'bedrooms': np.random.randint(1, 6, n_samples),
        'bathrooms': np.random.uniform(1, 4, n_samples).round(1),
        'age': np.random.randint(0, 50, n_samples),
        'location_score': np.random.uniform(1, 10, n_samples).round(1)
    }
    
    df = pd.DataFrame(data)
    
    df['price'] = (
        df['sqft'] * 150 +
        df['bedrooms'] * 10000 +
        df['bathrooms'] * 15000 +
        df['location_score'] * 20000 -
        df['age'] * 2000 +
        np.random.normal(0, 50000, n_samples)
    )
    
    missing_indices = np.random.choice(df.index, size=int(n_samples * 0.02), replace=False)
    df.loc[missing_indices, 'bathrooms'] = np.nan
    
    return df

def preprocess_data(df):
    print("Starting preprocessing...")
    
    print(f"Missing values before: {df.isnull().sum().sum()}")
    df['bathrooms'].fillna(df['bathrooms'].median(), inplace=True)
    print(f"Missing values after: {df.isnull().sum().sum()}")
    
    Q1 = df['price'].quantile(0.25)
    Q3 = df['price'].quantile(0.75)
    IQR = Q3 - Q1
    df = df[(df['price'] >= Q1 - 1.5 * IQR) & (df['price'] <= Q3 + 1.5 * IQR)]
    
    print(f"Dataset shape after cleaning: {df.shape}")
    return df

def train_model(X_train, y_train):
    print("\nTraining models...")
    
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    }
    
    best_model = None
    best_score = -np.inf
    best_name = ""
    
    for name, model in models.items():
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
        mean_cv_score = cv_scores.mean()
        
        print(f"{name} - CV R² Score: {mean_cv_score:.4f} (+/- {cv_scores.std():.4f})")
        
        if mean_cv_score > best_score:
            best_score = mean_cv_score
            best_model = model
            best_name = name
    
    print(f"\nBest model: {best_name}")
    best_model.fit(X_train, y_train)
    
    return best_model, best_name

def evaluate_model(model, X_train, y_train, X_test, y_test):
    print("\n" + "="*50)
    print("MODEL EVALUATION")
    print("="*50)
    
    train_pred = model.predict(X_train)
    train_r2 = r2_score(y_train, train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    train_mae = mean_absolute_error(y_train, train_pred)
    
    test_pred = model.predict(X_test)
    test_r2 = r2_score(y_test, test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    test_mae = mean_absolute_error(y_test, test_pred)
    
    print(f"\nTraining Metrics:")
    print(f"  R² Score: {train_r2:.4f}")
    print(f"  RMSE: ${train_rmse:,.2f}")
    print(f"  MAE: ${train_mae:,.2f}")
    
    print(f"\nTesting Metrics:")
    print(f"  R² Score: {test_r2:.4f}")
    print(f"  RMSE: ${test_rmse:,.2f}")
    print(f"  MAE: ${test_mae:,.2f}")
    
    overfit_diff = train_r2 - test_r2
    print(f"\nOverfitting Check:")
    print(f"  R² Difference: {overfit_diff:.4f}")
    if overfit_diff > 0.15:
        print("  ⚠️  Warning: Model may be overfitting!")
    else:
        print("  ✓ Model generalization looks good")
    
    print(f"\nSample Predictions (first 5):")
    for i in range(min(5, len(test_pred))):
        print(f"  Actual: ${y_test.iloc[i]:,.0f} | Predicted: ${test_pred[i]:,.0f}")
    
    return {
        'train_r2': train_r2,
        'test_r2': test_r2,
        'rmse': test_rmse,
        'mae': test_mae
    }

def save_model_and_scaler(model, scaler, feature_names, metrics):
    from datetime import datetime
    
    model_data = {
        'model': model,
        'scaler': scaler,
        'feature_names': feature_names,
        'metrics': metrics,
        'trained_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    joblib.dump(model_data, 'house_price_model.pkl')
    print("\n✓ Model saved as 'house_price_model.pkl'")

def load_model_and_scaler():
    try:
        model_data = joblib.load('house_price_model.pkl')
        print("✓ Model loaded successfully")
        return model_data
    except FileNotFoundError:
        print("❌ Model file not found. Please train the model first.")
        return None

def make_prediction(model_data, sqft, bedrooms, bathrooms, age, location_score):
    input_data = pd.DataFrame({
        'sqft': [sqft],
        'bedrooms': [bedrooms],
        'bathrooms': [bathrooms],
        'age': [age],
        'location_score': [location_score]
    })
    
    input_scaled = model_data['scaler'].transform(input_data)
    prediction = model_data['model'].predict(input_scaled)[0]
    
    return prediction

def interactive_prediction():
    print("\n" + "="*50)
    print("HOUSE PRICE PREDICTION")
    print("="*50)
    
    model_data = load_model_and_scaler()
    if not model_data:
        return
    
    try:
        print("\nEnter house details:")
        sqft = float(input("  Square footage (500-5000): "))
        bedrooms = int(input("  Number of bedrooms (1-5): "))
        bathrooms = float(input("  Number of bathrooms (1-4): "))
        age = int(input("  Age of house in years (0-50): "))
        location_score = float(input("  Location score (1-10): "))
        
        if not (500 <= sqft <= 5000):
            print("❌ Square footage should be between 500 and 5000")
            return
        if not (1 <= bedrooms <= 5):
            print("❌ Bedrooms should be between 1 and 5")
            return
        
        predicted_price = make_prediction(model_data, sqft, bedrooms, bathrooms, age, location_score)
        
        print(f"\n{'='*50}")
        print(f"PREDICTED PRICE: ${predicted_price:,.2f}")
        print(f"{'='*50}")
        
        print(f"\nModel Accuracy (Test R²): {model_data['metrics']['test_r2']:.4f}")
        print(f"Average Error (MAE): ${model_data['metrics']['mae']:,.2f}")
        
    except ValueError as e:
        print(f"❌ Invalid input: {e}")
    except Exception as e:
        print(f"❌ Error: {e}")

def create_flask_app():
    from flask import Flask, request, jsonify
    from flask_cors import CORS
    
    app = Flask(__name__)
    CORS(app)
    
    MODEL_DATA = load_model_and_scaler()
    
    @app.route('/health', methods=['GET'])
    def health():
        return jsonify({
            'status': 'healthy',
            'model_loaded': MODEL_DATA is not None,
            'model_accuracy': float(MODEL_DATA['metrics']['test_r2']) if MODEL_DATA else None
        })
    
    @app.route('/predict', methods=['POST'])
    def predict_api():
        try:
            data = request.get_json()
            
            required_fields = ['sqft', 'bedrooms', 'bathrooms', 'age', 'location_score']
            for field in required_fields:
                if field not in data:
                    return jsonify({'error': f'Missing field: {field}'}), 400
            
            prediction = make_prediction(
                MODEL_DATA,
                float(data['sqft']),
                int(data['bedrooms']),
                float(data['bathrooms']),
                int(data['age']),
                float(data['location_score'])
            )
            
            return jsonify({
                'predicted_price': float(prediction),
                'currency': 'USD',
                'model_accuracy': float(MODEL_DATA['metrics']['test_r2']),
                'input_values': data
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    return app

def run_flask_api():
    print("\n" + "="*70)
    print("STARTING FLASK API SERVER")
    print("="*70)
    print("\nAPI Endpoints:")
    print("  GET  http://localhost:5000/health  - Check API health")
    print("  POST http://localhost:5000/predict - Make prediction")
    print("\nExample curl command:")
    print('  curl -X POST http://localhost:5000/predict \\')
    print('    -H "Content-Type: application/json" \\')
    print('    -d \'{"sqft": 2000, "bedrooms": 3, "bathrooms": 2, "age": 5, "location_score": 7}\'')
    print("\nPress Ctrl+C to stop the server")
    print("="*70 + "\n")
    
    app = create_flask_app()
    app.run(debug=True, host='0.0.0.0', port=5000)

def run_streamlit():
    import streamlit as st
    
    st.set_page_config(page_title="House Price Predictor", page_icon="🏠", layout="wide")
    
    st.title("🏠 House Price Prediction")
    st.markdown("### Predict house prices using Machine Learning")
    
    model_data = load_model_and_scaler()
    
    if model_data is None:
        st.error("❌ Model not found! Please train the model first.")
        st.stop()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Model Accuracy (R²)", f"{model_data['metrics']['test_r2']:.4f}")
    with col2:
        st.metric("Average Error", f"${model_data['metrics']['mae']:,.0f}")
    with col3:
        st.metric("Trained On", model_data['trained_date'])
    
    st.markdown("---")
    
    st.subheader("Enter House Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        sqft = st.number_input("Square Footage", min_value=500, max_value=5000, value=2000, step=100)
        bedrooms = st.number_input("Bedrooms", min_value=1, max_value=5, value=3, step=1)
        bathrooms = st.number_input("Bathrooms", min_value=1.0, max_value=4.0, value=2.0, step=0.5)
    
    with col2:
        age = st.number_input("Age (years)", min_value=0, max_value=50, value=10, step=1)
        location_score = st.slider("Location Score", min_value=1.0, max_value=10.0, value=7.0, step=0.1)
        st.caption("1 = Poor location, 10 = Excellent location")
    
    if st.button("🔮 Predict Price", type="primary", use_container_width=True):
        with st.spinner("Making prediction..."):
            try:
                predicted_price = make_prediction(
                    model_data, sqft, bedrooms, bathrooms, age, location_score
                )
                
                st.success("Prediction Complete!")
                
                st.markdown("### Predicted Price")
                st.markdown(f"## ${predicted_price:,.2f}")
                
                error_margin = model_data['metrics']['mae']
                lower_bound = predicted_price - error_margin
                upper_bound = predicted_price + error_margin
                
                st.info(f"📊 Estimated Range: ${lower_bound:,.2f} - ${upper_bound:,.2f}")
                
                st.markdown("---")
                st.subheader("Input Summary")
                summary_df = pd.DataFrame({
                    'Feature': ['Square Footage', 'Bedrooms', 'Bathrooms', 'Age', 'Location Score'],
                    'Value': [f"{sqft} sqft", bedrooms, bathrooms, f"{age} years", location_score]
                })
                st.dataframe(summary_df, hide_index=True, use_container_width=True)
                
            except Exception as e:
                st.error(f"❌ Error making prediction: {e}")
    
    with st.sidebar:
        st.header("About")
        st.markdown("""
        This app uses Machine Learning to predict house prices based on:
        - Square footage
        - Number of bedrooms
        - Number of bathrooms
        - Age of the house
        - Location quality score
        """)
        
        st.markdown("---")
        st.header("Model Info")
        st.markdown(f"**Algorithm:** Random Forest")
        st.markdown(f"**Features:** {len(model_data['feature_names'])}")
        st.markdown(f"**Trained:** {model_data['trained_date']}")

def generate_deployment_files():
    requirements = """pandas==2.0.0
numpy==1.24.0
scikit-learn==1.3.0
joblib==1.3.0
flask==3.0.0
flask-cors==4.0.0
streamlit==1.28.0
gunicorn==21.2.0
"""
    
    with open('requirements.txt', 'w') as f:
        f.write(requirements)
    print("✓ Created requirements.txt")
    
    procfile = """web: gunicorn app:app
"""
    
    with open('Procfile', 'w') as f:
        f.write(procfile)
    print("✓ Created Procfile")
    
    readme = """# House Price Prediction ML Project

## Quick Start

### Install Dependencies
```bash
pip install pandas numpy scikit-learn joblib flask flask-cors streamlit
```

### Train Model
```bash
python housepriceml.py
# Select option 1
```

### Run Streamlit Web App
```bash
streamlit run housepriceml.py
```

### Run Flask API
```bash
python housepriceml.py
# Select option 4
```
"""
    
    with open('README.md', 'w') as f:
        f.write(readme)
    print("✓ Created README.md")
    
    print("\n✅ All deployment files created successfully!")

def train_model_flow():
    print("\n" + "="*50)
    print("DATA PREPARATION")
    print("="*50)
    df = generate_sample_data(n_samples=1000)
    print(f"Generated {len(df)} samples")
    print(f"\nDataset info:")
    print(df.describe())
    
    df = preprocess_data(df)
    
    X = df[['sqft', 'bedrooms', 'bathrooms', 'age', 'location_score']]
    y = df['price']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model, model_name = train_model(X_train_scaled, y_train)
    
    metrics = evaluate_model(model, X_train_scaled, y_train, X_test_scaled, y_test)
    
    save_model_and_scaler(model, scaler, X.columns.tolist(), metrics)

def main():
    print("="*70)
    print("HOUSE PRICE PREDICTION - ML PROJECT")
    print("="*70)
    
    print("\nSelect Mode:")
    print("1. Train new model")
    print("2. Make prediction (command line)")
    print("3. Train + Predict")
    print("4. Run Flask API")
    print("5. Run Streamlit Web App (use: streamlit run housepriceml.py)")
    print("6. Generate deployment files")
    
    choice = input("\nEnter choice (1-6): ").strip()
    
    if choice == '1':
        train_model_flow()
    
    elif choice == '2':
        interactive_prediction()
    
    elif choice == '3':
        train_model_flow()
        interactive_prediction()
    
    elif choice == '4':
        run_flask_api()
    
    elif choice == '5':
        print("\n" + "="*70)
        print("To run Streamlit, use this command:")
        print("  streamlit run housepriceml.py")
        print("="*70)
    
    elif choice == '6':
        generate_deployment_files()
    
    else:
        print("Invalid choice. Please run again and select 1-6.")

if __name__ == "__main__":
    if 'streamlit' in sys.modules:
        run_streamlit()
    else:
        main()