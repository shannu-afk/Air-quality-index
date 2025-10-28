from flask import Flask
from flask import request, render_template, url_for
import pickle
import pandas as pd
import logging
import numpy as np

# Simple logging to stdout for debugging predictions
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_aqi_category(pm25_value):
    """Convert PM2.5 value to AQI category and description."""
    if pm25_value <= 12.0:
        return "Good", "Air quality is satisfactory, and air pollution poses little or no risk."
    elif pm25_value <= 35.4:
        return "Moderate", "Air quality is acceptable. However, there may be a risk for some people, particularly those who are unusually sensitive to air pollution."
    elif pm25_value <= 55.4:
        return "Unhealthy for Sensitive Groups", "Members of sensitive groups may experience health effects. The general public is less likely to be affected."
    elif pm25_value <= 150.4:
        return "Unhealthy", "Everyone may begin to experience health effects; members of sensitive groups may experience more serious health effects."
    elif pm25_value <= 250.4:
        return "Very Unhealthy", "Health alert: everyone may experience more serious health effects."
    else:
        return "Hazardous", "Health warnings of emergency conditions. The entire population is more likely to be affected."

def validate_and_convert_input(form_data):
    """Validate and convert form inputs to floats, with bounds checking."""
    required_fields = ['T', 'TM', 'Tm', 'SLP', 'H', 'VV', 'V', 'VM']
    values = []
    
    # Log raw form data for debugging
    logger.info("Raw form data: %s", {k: form_data.get(k, 'MISSING') for k in required_fields})
    
    for field in required_fields:
        try:
            val = float(form_data[field])
            # Basic sanity checks
            if not np.isfinite(val):  # catches inf, -inf, nan
                raise ValueError(f"{field} has invalid value: {val}")
            values.append(val)
        except KeyError:
            logger.error(f"Missing required field: {field}")
            raise ValueError(f"Missing required field: {field}")
        except ValueError as e:
            logger.error(f"Invalid value for {field}: {form_data[field]}")
            raise ValueError(f"Invalid value for {field}: {form_data[field]}")
    
    logger.info("Converted values: %s", values)
    return values

app=Flask(__name__)

try:
    model = pickle.load(open('tree_gridcv.pkl', 'rb'))
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
	AQI_predict = None
	if request.method == 'POST':
		try:
			if model is None:
				raise ValueError("Model not loaded. Please check the model file.")

			# Validate and convert all inputs
			input_values = validate_and_convert_input(request.form)
			
			# Build DataFrame with validated values
			cols = ['T', 'TM', 'Tm', 'SLP', 'H', 'VV', 'V', 'VM']
			df_input = pd.DataFrame([input_values], columns=cols)
			
			# Make prediction
			AQI_predict = model.predict(df_input)
			
			# Verify prediction is valid
			if not np.isfinite(AQI_predict[0]):
				raise ValueError(f"Model returned invalid prediction: {AQI_predict[0]}")
			
			# Get AQI category and description
			category, description = get_aqi_category(AQI_predict[0])
			logger.info(f"Valid prediction made: {AQI_predict[0]} ({category})")

		except Exception as e:
			error_msg = f"Prediction error: {str(e)}"
			logger.error(error_msg)
			AQI_predict = [error_msg]
			return render_template('result.html', prediction=AQI_predict, debug_info={
				'error': str(e),
				'input_data': dict(request.form)
			})

		# Store debug info to display in template
		debug_info = {
			'input_values': input_values,
			'prediction': AQI_predict[0] if AQI_predict is not None else None,
			'raw_form_data': dict(request.form),
			'aqi_category': get_aqi_category(AQI_predict[0]) if AQI_predict is not None else None
		}

	return render_template('result.html', prediction=AQI_predict, debug_info=debug_info)

if __name__=='__main__':
	app.run()
