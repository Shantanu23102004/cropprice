# from django.shortcuts import render
# from django.http import JsonResponse
# import pandas as pd
# import joblib
# import os

# # Load the model and label encoders
# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# MODEL_PATH = os.path.join(BASE_DIR, 'crop_price_prediction_model.pkl')
# ENCODERS_PATH = os.path.join(BASE_DIR, 'label_encoders.pkl')

# rf_regressor = joblib.load(MODEL_PATH)
# label_encoders = joblib.load(ENCODERS_PATH)

# def predict_price(request):
#     if request.method == 'GET':
#         state = request.GET.get('state')
#         district = request.GET.get('district')
#         market = request.GET.get('market')
#         commodity = request.GET.get('commodity')
#         year = int(request.GET.get('year'))
#         month = int(request.GET.get('month'))
#         day = int(request.GET.get('day'))

#         try:
#             # Encode categorical inputs
#             state_encoded = label_encoders['State'].transform([state])[0]
#             district_encoded = label_encoders['District'].transform([district])[0]
#             market_encoded = label_encoders['Market'].transform([market])[0]
#             commodity_encoded = label_encoders['Commodity'].transform([commodity])[0]

#             # Prepare input dataframe
#             user_input = pd.DataFrame({
#                 'State': [state_encoded],
#                 'District': [district_encoded],
#                 'Market': [market_encoded],
#                 'Commodity': [commodity_encoded],
#                 'Variety': [0],
#                 'Grade': [0],
#                 'Commodity_Code': [0],
#                 'Year': [year],
#                 'Month': [month],
#                 'Day': [day]
#             })

#             # Predict price
#             predicted_price = rf_regressor.predict(user_input)[0]
#             return JsonResponse({'predicted_price': predicted_price})

#         except Exception as e:
#             return JsonResponse({'error': str(e)}, status=500)

#     return JsonResponse({'error': 'Invalid request'}, status=400)


# def predict_price_page(request):
#     return render(request, 'predictPrice.html')

# from django.shortcuts import render
# from django.http import JsonResponse
# import pandas as pd
# import joblib
# import os
# import traceback  # For detailed error messages

# # Load the model and label encoders
# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# MODEL_PATH = os.path.join(BASE_DIR,  'crop_price_prediction_model.pkl')
# ENCODERS_PATH = os.path.join(BASE_DIR, 'label_encoders.pkl')

# try:
#     rf_regressor = joblib.load(MODEL_PATH)
#     label_encoders = joblib.load(ENCODERS_PATH)
# except Exception as e:
#     print("Error loading model:", e)

# def predict_price(request):
#     if request.method == 'GET':
#         try:
#             state = request.GET.get('state')
#             district = request.GET.get('district')
#             market = request.GET.get('market')
#             commodity = request.GET.get('commodity')
#             year = int(request.GET.get('year'))
#             month = int(request.GET.get('month'))
#             day = int(request.GET.get('day'))

#             print(f"Received input: {state}, {district}, {market}, {commodity}, {year}-{month}-{day}")

#             # Check if label encoders are loaded
#             if not isinstance(label_encoders, dict):
#                 return JsonResponse({'error': 'Label encoders not loaded properly'}, status=500)

#             # Encode categorical inputs
#             try:
#                 state_encoded = label_encoders['State'].transform([state])[0]
#                 district_encoded = label_encoders['District'].transform([district])[0]
#                 market_encoded = label_encoders['Market'].transform([market])[0]
#                 commodity_encoded = label_encoders['Commodity'].transform([commodity])[0]
#             except KeyError as e:
#                 return JsonResponse({'error': f'Invalid input: {e}'}, status=400)

#             # Prepare input dataframe
#             user_input = pd.DataFrame({
#                 'State': [state_encoded],
#                 'District': [district_encoded],
#                 'Market': [market_encoded],
#                 'Commodity': [commodity_encoded],
#                 'Variety': [0],
#                 'Grade': [0],
#                 'Commodity_Code': [0],
#                 'Year': [year],
#                 'Month': [month],
#                 'Day': [day]
#             })

#             print("User input dataframe:", user_input)

#             # Predict price
#             predicted_price = rf_regressor.predict(user_input)[0]
#             print(f"Predicted price: {predicted_price}")

#             return JsonResponse({'predicted_price': predicted_price})

#         except Exception as e:
#             error_details = traceback.format_exc()
#             print("Prediction error:", error_details)
#             return JsonResponse({'error': str(e), 'details': error_details}, status=500)

#     return JsonResponse({'error': 'Invalid request'}, status=400)

# def predict_price_page(request):
#     return render(request, 'predictPrice.html')

# =======================
from django.shortcuts import render
from django.http import JsonResponse
import pandas as pd
import joblib
import os
import traceback
import gdown

# Initialize with None
rf_regressor = None
label_encoders = None

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_URL= "https://drive.google.com/uc?id=1AX5aSaiSdaBDiL8y2hU1e1N-x-pO-5R1"
MODEL_PATH = os.path.join(BASE_DIR, '../crop_price_prediction_model.pkl')
ENCODERS_PATH = os.path.join(BASE_DIR, 'label_encoders.pkl')

# Debug: Print paths
print(f"Looking for model at: {MODEL_PATH}")
print(f"Looking for encoders at: {ENCODERS_PATH}")

if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
    
try:
    rf_regressor = joblib.load(MODEL_PATH)
    label_encoders = joblib.load(ENCODERS_PATH)
    print("Model and encoders loaded successfully!")
    # Debug: Print available categories
    print("\nAvailable categories in encoders:")
    print("States:", list(label_encoders['State'].classes_)[:5], "...")
    print("Districts:", list(label_encoders['District'].classes_)[:5], "...")
    print("Commodities:", list(label_encoders['Commodity'].classes_)[:5], "...")
except Exception as e:
    print(f"\nERROR LOADING FILES: {str(e)}")
    traceback.print_exc()

def predict_price(request):
    if request.method == 'GET':
        try:
            # Check if model loaded
            if rf_regressor is None or label_encoders is None:
                return JsonResponse({
                    'error': 'Prediction service unavailable (model not loaded)',
                    'debug': {
                        'model_loaded': rf_regressor is not None,
                        'encoders_loaded': label_encoders is not None
                    }
                }, status=503)

            # Get parameters
            params = {
                'state': request.GET.get('state'),
                'district': request.GET.get('district'),
                'market': request.GET.get('market'),
                'commodity': request.GET.get('commodity'),
                'year': int(request.GET.get('year')),
                'month': int(request.GET.get('month')),
                'day': int(request.GET.get('day'))
            }
            
            print("\nReceived prediction request with:")
            for k, v in params.items():
                print(f"- {k}: {v} ({type(v)})")

            # Verify commodity exists in encoder
            if params['commodity'] not in label_encoders['Commodity'].classes_:
                return JsonResponse({
                    'error': f"Commodity '{params['commodity']}' not found",
                    'available_commodities': list(label_encoders['Commodity'].classes_)
                }, status=400)

            # Encode inputs
            encoded = {
                'State': label_encoders['State'].transform([params['state']])[0],
                'District': label_encoders['District'].transform([params['district']])[0],
                'Market': label_encoders['Market'].transform([params['market']])[0],
                'Commodity': label_encoders['Commodity'].transform([params['commodity']])[0]
            }

            # Create input DataFrame
            input_data = pd.DataFrame({
                'State': [encoded['State']],
                'District': [encoded['District']],
                'Market': [encoded['Market']],
                'Commodity': [encoded['Commodity']],
                'Variety': [0],  # Default values
                'Grade': [0],
                'Commodity_Code': [0],
                'Year': [params['year']],
                'Month': [params['month']],
                'Day': [params['day']]
            })

            print("\nInput data for prediction:")
            print(input_data)

            # Make prediction
            prediction = rf_regressor.predict(input_data)[0]
            print(f"\nPrediction result: {prediction}")

            return JsonResponse({
                'predicted_price': float(prediction),
                'units': '₹/Quintal'  # Assuming price is in INR per quintal
            })

        except Exception as e:
            print(f"\nPrediction failed: {str(e)}")
            traceback.print_exc()
            return JsonResponse({
                'error': 'Prediction failed',
                'details': str(e)
            }, status=500)

    return JsonResponse({'error': 'Method not allowed'}, status=405)

def predict_price_page(request):
    return render(request, 'predictPrice.html')
# =================================
# import requests
# from datetime import datetime, timedelta
# import random
# from django.shortcuts import render
# from django.http import JsonResponse
# import pandas as pd
# import joblib
# import os
# import traceback

# # Initialize model and encoders
# rf_regressor = None
# label_encoders = None

# # Load model and encoders
# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# MODEL_PATH = os.path.join(BASE_DIR, 'crop_price_prediction_model.pkl')
# ENCODERS_PATH = os.path.join(BASE_DIR, 'label_encoders.pkl')

# try:
#     rf_regressor = joblib.load(MODEL_PATH)
#     label_encoders = joblib.load(ENCODERS_PATH)
#     print("Model and label encoders loaded successfully!")
# except Exception as e:
#     print(f"Error loading model or encoders: {e}")

# API_KEY = "579b464db66ec23bdd0000017f5d751b58db42284a90bb19d5591743"

# def fetch_price_data(state, district, market, commodity, date):
#     base_url = "https://api.data.gov.in/resource/35985678-0d79-46b4-9ed6-6f13308a1d24"
#     params = {
#         "api-key": API_KEY,
#         "format": "json",
#         "filters[State]": state,
#         "filters[District]": district,
#         "filters[Market]": market,
#         "filters[Commodity]": commodity,
#         "filters[Arrival_Date]": date,
#         "limit": 1
#     }
    
#     try:
#         response = requests.get(base_url, params=params)
#         if response.status_code == 200:
#             data = response.json()
#             if "records" in data and len(data["records"]) > 0:
#                 return float(data["records"][0].get("modal_price", 0))
#     except Exception as e:
#         print(f"Error fetching price data: {e}")
#     return None

# def get_chart_data(state, district, market, commodity, year):
#     chart_data = {
#         'current_year': [None] * 7,  # For 7 months
#         'previous_year': [None] * 7,
#         'current_price': None,
#         'last_year_price': None
#     }
    
#     current_month = datetime.now().month if year == datetime.now().year else 7
    
#     # Get current year data
#     for month in range(1, min(current_month + 1, 8)):  # Limit to 7 months
#         date = f"{year}-{month:02d}-01"
#         price = fetch_price_data(state, district, market, commodity, date)
#         chart_data['current_year'][month-1] = price
    
#     # Get current price (most recent available)
#     params = {
#         "api-key": API_KEY,
#         "format": "json",
#         "filters[State]": state,
#         "filters[District]": district,
#         "filters[Market]": market,
#         "filters[Commodity]": commodity,
#         "limit": 1,
#         "sort[Arrival_Date]": "desc"
#     }
    
#     try:
#         response = requests.get("https://api.data.gov.in/resource/35985678-0d79-46b4-9ed6-6f13308a1d24", params=params)
#         if response.status_code == 200:
#             data = response.json()
#             if "records" in data and len(data["records"]) > 0:
#                 chart_data['current_price'] = float(data["records"][0].get("modal_price", 0))
                
#                 # Get last year price (same month)
#                 arrival_date = data["records"][0].get("arrival_date", "")
#                 if arrival_date:
#                     month = arrival_date.split('-')[1]
#                     prev_year_date = f"{year-1}-{month}-01"
#                     prev_price = fetch_price_data(state, district, market, commodity, prev_year_date)
#                     chart_data['last_year_price'] = prev_price
#     except Exception as e:
#         print(f"Error fetching current price: {e}")
    
#     # Get previous year data for same months
#     prev_year = year - 1
#     for month in range(1, min(current_month + 1, 8)):  # Limit to 7 months
#         date = f"{prev_year}-{month:02d}-01"
#         price = fetch_price_data(state, district, market, commodity, date)
#         chart_data['previous_year'][month-1] = price
    
#     return chart_data

# def predict_price_page(request):
#     context = {
#         'chart_data': {
#             'labels': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul'],
#             'current_year': [None]*7,
#             'previous_year': [None]*7,
#             'current_price': None,
#             'last_year_price': None
#         }
#     }
    
#     if request.method == 'GET' and all(k in request.GET for k in ['state', 'district', 'market', 'commodity']):
#         state = request.GET.get('state')
#         district = request.GET.get('district')
#         market = request.GET.get('market')
#         commodity = request.GET.get('commodity')
#         year = int(request.GET.get('year', datetime.now().year))
        
#         chart_data = get_chart_data(state, district, market, commodity, year)
#         context['chart_data'].update(chart_data)
    
#     return render(request, 'predictPrice.html', context)

# def predict_price(request):
#     if request.method == 'GET':
#         try:
#             if rf_regressor is None or label_encoders is None:
#                 return JsonResponse({'error': 'Model not loaded'}, status=500)

#             state = request.GET.get('state')
#             district = request.GET.get('district')
#             market = request.GET.get('market')
#             commodity = request.GET.get('commodity')
#             year = int(request.GET.get('year'))
#             month = int(request.GET.get('month'))
#             day = int(request.GET.get('day'))

#             # Get chart data
#             chart_data = get_chart_data(state, district, market, commodity, year)
            
#             # Make prediction
#             state_encoded = label_encoders['State'].transform([state])[0]
#             district_encoded = label_encoders['District'].transform([district])[0]
#             market_encoded = label_encoders['Market'].transform([market])[0]
#             commodity_encoded = label_encoders['Commodity'].transform([commodity])[0]

#             user_input = pd.DataFrame({
#                 'State': [state_encoded],
#                 'District': [district_encoded],
#                 'Market': [market_encoded],
#                 'Commodity': [commodity_encoded],
#                 'Variety': [0],
#                 'Grade': [0],
#                 'Commodity_Code': [0],
#                 'Year': [year],
#                 'Month': [month],
#                 'Day': [day]
#             })

#             predicted_price = rf_regressor.predict(user_input)[0]

#             return JsonResponse({
#                 'predicted_price': predicted_price,
#                 'chart_data': chart_data
#             })

#         except Exception as e:
#             error_details = traceback.format_exc()
#             print("Prediction error:", error_details)
#             return JsonResponse({'error': str(e), 'details': error_details}, status=500)

#     return JsonResponse({'error': 'Invalid request'}, status=400)