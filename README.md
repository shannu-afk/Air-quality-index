# Air Quality Index Prediction 

A Flask-based web application that predicts Air Quality Index (AQI) using machine learning. The model takes various weather parameters as input and predicts the AQI value based on PM 2.5 measurements.

## Features

- **Web Interface**: User-friendly web form for inputting weather parameters
- **Machine Learning Prediction**: Uses a trained model to predict AQI from weather data
- **Error Handling**: Robust error handling for model loading and prediction failures
- **Responsive Design**: Bootstrap-based responsive UI

## Input Parameters

The model requires the following weather parameters:

- **T**: Average Temperature (°C)
- **TM**: Maximum Temperature (°C)
- **Tm**: Minimum Temperature (°C)
- **SLP**: Atmospheric pressure at sea level (hectopascal)
- **H**: Average relative humidity (%)
- **VV**: Average visibility (Km)
- **V**: Average wind speed (Km/h)
- **VM**: Maximum sustained wind speed (Km/h)

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd Air-Quality-Index
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Ensure the model file `tree_gridcv.pkl` is present in the root directory.

## Usage

1. Run the application locally:
   ```bash
   python app.py
   ```

2. Open your browser and navigate to `http://localhost:5000`

3. Enter the weather parameters in the form and click "Predict" to get the AQI value.


The app is configured with:
- `runtime.txt`: Specifies Python version 3.8.10
- `requirements.txt`: Lists all dependencies

## Project Structure

```
Air-Quality-Index-prediction_with_deployment/
├── app.py                    # Main Flask application
├── requirements.txt          # Python dependencies
├── Procfile                  # Heroku deployment configuration
├── runtime.txt              # Python runtime version
├── tree_gridcv.pkl          # Trained ML model
├── templates/               # HTML templates
│   ├── layout.html
│   ├── home.html
│   └── result.html
├── static/                  # Static files (CSS, images)
│   ├── css/
│   └── images/
├── Dataset/                 # Dataset files
├── ML_Algorithms/           # Jupyter notebooks for model training
└── ML_Applied_Word/         # Additional ML processing code
```

## Model Training

The machine learning model was trained using various algorithms including:
- Decision Tree
- Extra Trees Regressor
- XGBoost Regressor
- Support Vector Regressor

The trained model (`tree_gridcv.pkl`) is included in the repository. For retraining or exploring the training process, refer to the notebooks in the `ML_Algorithms/` directory.

## Dependencies

- Flask==2.0.0
- scikit-learn==0.22.2.post1
- numpy==1.20.3
- gunicorn==20.1.0
- Bootstrap 4.0.0 (via CDN)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is open source. Please check the license file for more details.

## Acknowledgments

- Dataset source: Air Quality Index data
- Model training algorithms implemented in Jupyter notebooks
