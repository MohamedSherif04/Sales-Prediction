
from flask import Flask, request, render_template_string
import pandas as pd
import joblib
from datetime import datetime

# تحميل الملفات
model_columns = joblib.load('columns.pkl')
models = {
    'XGBoost': joblib.load('xgb_model.pkl'),
    'Random Forest': joblib.load('rf_model.pkl'),
    'Gradient Boosting': joblib.load('gbr_model.pkl')
}

# القيم المسموح بها من الداتا الأصلية
allowed_sales_channels = ['In-Store', 'Online', 'Distributor', 'Wholesale']
allowed_warehouses = ['WARE-UHY1004', 'WARE-NMK1003', 'WARE-PUJ1005', 'WARE-XYS1001', 'WARE-MKL1006', 'WARE-NBV1002']
limits = {
    '_ProductID': (1, 47),
    'Order Quantity': (1, 8),
    'Discount Applied': (0.05, 0.4),
    'Unit Cost': (68.68, 5498.56)
}

INPUT_COLUMNS = ['_ProductID', 'Order Quantity', 'Discount Applied', 'Unit Cost', 'Sales Channel', 'WarehouseCode', 'OrderDate']

HTML = """<!DOCTYPE html>
<html><head><title>🧠 Predict Unit Price</title><style>
body { font-family: Arial; max-width: 700px; margin: 50px auto; padding: 20px; background: #f9f9f9; border-radius: 10px; }
h2 { text-align: center; }
form { display: flex; flex-direction: column; }
label { margin: 10px 0 5px; font-weight: bold; }
input, select { padding: 8px; border-radius: 5px; border: 1px solid #ccc; }
.submit-btn { margin-top: 20px; padding: 10px; background: #007bff; color: white; border: none; border-radius: 5px; cursor: pointer; }
.submit-btn:hover { background: #0056b3; }
.result { margin-top: 30px; padding: 15px; background: #e8f0fe; border-left: 5px solid #2196f3; }
</style></head><body>
<h2>🧠 Predict Unit Price</h2>
<form method="post">
<label>Choose Model:</label><select name="Model" required>
{% for model in models %}<option value="{{ model }}">{{ model }}</option>{% endfor %}
</select>
{% for col in input_columns %}
<label>{{ col.replace('_', ' ').title() }}:</label>
{% if col == 'OrderDate' %}
<input type="date" name="{{ col }}" required>
{% elif col in ['Sales Channel', 'WarehouseCode'] %}
<input type="text" name="{{ col }}" required>
{% else %}
<input type="number" step="any" name="{{ col }}" required>
{% endif %}
{% endfor %}
<button type="submit" class="submit-btn">Predict</button></form>
{% if prediction %}
<div class="result">Predicted Unit Price ({{ chosen_model }}): <strong>${{ prediction }}</strong></div>
{% endif %}
</body></html>"""

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    chosen_model = None

    if request.method == 'POST':
        try:
            form = request.form
            chosen_model = form['Model']
            order_date = pd.to_datetime(form['OrderDate'])

            if form['Sales Channel'] not in allowed_sales_channels:
                raise ValueError("❌ Invalid Sales Channel")
            if form['WarehouseCode'] not in allowed_warehouses:
                raise ValueError("❌ Invalid Warehouse Code")

            for field in ['_ProductID', 'Order Quantity', 'Discount Applied', 'Unit Cost']:
                val = float(form[field])
                min_val, max_val = limits[field]
                if not (min_val <= val <= max_val):
                    raise ValueError(f"❌ {field} out of range ({min_val} to {max_val})")

            features = {
                '_ProductID': float(form['_ProductID']),
                'Order Quantity': float(form['Order Quantity']),
                'Discount Applied': float(form['Discount Applied']),
                'Unit Cost': float(form['Unit Cost']),
                'OrderYear': order_date.year,
                'OrderMonth': order_date.month,
                'OrderDay': order_date.day,
                'OrderDayOfWeek': order_date.dayofweek,
                'IsWeekend': int(order_date.dayofweek in [5, 6]),
                'OrderWeek': order_date.isocalendar().week,
                'Rolling_Avg_7': 0,
                'Rolling_Avg_14': 0,
                'Rolling_Avg_30': 0,
                'Lag_1': 0,
                'Lag_2': 0,
                'Lag_6': 0,
            }

            for col in model_columns:
                if col.startswith('Sales Channel_'):
                    features[col] = 1 if col.endswith(form['Sales Channel']) else 0
                elif col.startswith('WarehouseCode_'):
                    features[col] = 1 if col.endswith(form['WarehouseCode']) else 0

            for col in model_columns:
                if col not in features:
                    features[col] = 0

            df_input = pd.DataFrame([features])[model_columns]
            model = models[chosen_model]
            pred = model.predict(df_input)
            prediction = round(pred[0], 2)

        except Exception as e:
            prediction = str(e)

    return render_template_string(HTML, prediction=prediction, models=models.keys(), input_columns=INPUT_COLUMNS, chosen_model=chosen_model)

if __name__ == '__main__':
    app.run(debug=True)
