from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import os
from ensemble_engine import (
    run_ensemble_comparison,
    analyze_bias_variance,
    analyze_overfitting_underfitting,
    analyze_regularization
)

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

@app.route('/api/ensemble/compare', methods=['GET'])
def api_ensemble():
    res = run_ensemble_comparison()
    return jsonify({"success": True, "data": res})

@app.route('/api/bias-variance', methods=['GET'])
def api_bias_variance():
    res = analyze_bias_variance()
    return jsonify({"success": True, "data": res})

@app.route('/api/overfitting', methods=['GET'])
def api_overfitting():
    res = analyze_overfitting_underfitting()
    return jsonify({"success": True, "data": res})

@app.route('/api/regularization', methods=['GET'])
def api_regularization():
    res = analyze_regularization()
    return jsonify({"success": True, "data": res})

if __name__ == '__main__':
    print("=" * 60)
    print("[WEEK 13-14] Ensemble & Regularization Studio")
    print("[SERVER] Starting server at http://127.0.0.1:5006")
    print("=" * 60)
    app.run(debug=True, host='0.0.0.0', port=5006)
