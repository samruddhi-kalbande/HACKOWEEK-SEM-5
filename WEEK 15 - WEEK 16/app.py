import os
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from nn_deployment_engine import (
    benchmark_deployment_serialization,
    deploy_predict,
    get_activation_curves,
    simulate_optimizers,
    train_neural_network_experiment,
    compute_decision_boundary
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

@app.route('/api/deployment/benchmark', methods=['GET'])
def api_deployment_benchmark():
    try:
        res = benchmark_deployment_serialization()
        return jsonify({"success": True, "data": res})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/deployment/predict', methods=['POST'])
def api_deployment_predict():
    try:
        payload = request.get_json(force=True) or {}
        serializer = payload.get('serializer', 'joblib')
        features = payload.get('features', {})

        # Default fallbacks if empty
        default_features = {
            'Age': 20,
            'StudyHoursPerWeek': 15.0,
            'AttendanceRate': 85.0,
            'MathScore': 80.0,
            'ReadingScore': 78.0,
            'WritingScore': 82.0,
            'Gender': 'Female',
            'Department': 'Computer Science'
        }
        for k, v in default_features.items():
            if k not in features:
                features[k] = v

        res = deploy_predict(features, serializer=serializer)
        return jsonify({"success": True, "data": res})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/activations', methods=['GET'])
def api_activations():
    try:
        res = get_activation_curves()
        return jsonify({"success": True, "data": res})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/optimizers/simulate', methods=['GET'])
def api_optimizers_simulate():
    try:
        res = simulate_optimizers()
        return jsonify({"success": True, "data": res})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/neural-net/train', methods=['POST', 'GET'])
def api_train_nn():
    try:
        if request.method == 'POST':
            payload = request.get_json(force=True) or {}
            epochs = int(payload.get('epochs', 100))
            lr = float(payload.get('learning_rate', 0.03))
            activation = payload.get('activation', 'relu')
            optimizer = payload.get('optimizer', 'adam')
            hidden_units = int(payload.get('hidden_units', 16))
        else:
            epochs = int(request.args.get('epochs', 100))
            lr = float(request.args.get('learning_rate', 0.03))
            activation = request.args.get('activation', 'relu')
            optimizer = request.args.get('optimizer', 'adam')
            hidden_units = int(request.args.get('hidden_units', 16))

        res = train_neural_network_experiment(
            epochs=epochs,
            lr=lr,
            activation=activation,
            optimizer=optimizer,
            hidden_units=hidden_units
        )
        return jsonify({"success": True, "data": res})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/neural-net/decision-boundary', methods=['GET'])
def api_decision_boundary():
    try:
        res = compute_decision_boundary()
        return jsonify({"success": True, "data": res})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == '__main__':
    print("=" * 65)
    print("[WEEK 15-16] Neural Networks from Scratch & Model Deployment Studio")
    print("[SERVER] Running at: http://127.0.0.1:5007")
    print("=" * 65)
    app.run(debug=True, host='0.0.0.0', port=5007)
