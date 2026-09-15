from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import os
from ml_engine import train_and_benchmark, predict_custom_student

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PUBLIC_DIR = os.path.join(BASE_DIR, 'public')

app = Flask(__name__, static_folder=PUBLIC_DIR, static_url_path='')
CORS(app)

# ─── Static Routing ──────────────────────────────────────────────────────────

@app.route('/')
def index():
    return send_from_directory(PUBLIC_DIR, 'index.html')

@app.route('/<path:path>')
def static_proxy(path):
    if os.path.exists(os.path.join(PUBLIC_DIR, path)):
        return send_from_directory(PUBLIC_DIR, path)
    return send_from_directory(PUBLIC_DIR, 'index.html')

# ─── API Routes ──────────────────────────────────────────────────────────────

@app.route('/api/models/benchmarks', methods=['GET'])
def get_benchmarks():
    ridge_alpha = float(request.args.get('ridge_alpha', 1.0))
    lasso_alpha = float(request.args.get('lasso_alpha', 0.05))
    knn_k = int(request.args.get('knn_k', 3))
    
    benchmarks = train_and_benchmark(ridge_alpha, lasso_alpha, knn_k)
    return jsonify({"success": True, "benchmarks": benchmarks})

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.get_json() or {}
    
    study_hours = float(data.get('study_hours', 18.0))
    attendance = float(data.get('attendance', 90.0))
    math = float(data.get('math', 85.0))
    reading = float(data.get('reading', 85.0))
    writing = float(data.get('writing', 85.0))
    
    ridge_alpha = float(data.get('ridge_alpha', 1.0))
    lasso_alpha = float(data.get('lasso_alpha', 0.05))
    knn_k = int(data.get('knn_k', 3))

    res = predict_custom_student(
        study_hours, attendance, math, reading, writing,
        ridge_alpha, lasso_alpha, knn_k
    )
    return jsonify({"success": True, "prediction": res})

if __name__ == '__main__':
    print("=" * 60)
    print("[WEEK 7-8] Supervised ML: Regression & Classification Studio")
    print("[SERVER] Starting server at http://127.0.0.1:5003")
    print("=" * 60)
    app.run(debug=True, host='0.0.0.0', port=5003)
