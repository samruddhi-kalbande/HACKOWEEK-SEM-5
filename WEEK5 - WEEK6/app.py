from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import os
from math_engine import (
    vector_operations,
    apply_2d_transformation,
    compute_eigen_analysis,
    run_gradient_descent,
    compute_chain_rule_step
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

@app.route('/api/linalg/vector-ops', methods=['POST'])
def api_vector_ops():
    data = request.get_json() or {}
    u = data.get('u', [3, 4])
    v = data.get('v', [1, 2])
    res = vector_operations(u, v)
    return jsonify({"success": True, "result": res})

@app.route('/api/linalg/matrix-transform', methods=['POST'])
def api_matrix_transform():
    data = request.get_json() or {}
    matrix = data.get('matrix', [[1, 0.5], [0, 1]])
    res = apply_2d_transformation(matrix)
    return jsonify({"success": True, "result": res})

@app.route('/api/linalg/eigen-analysis', methods=['GET'])
def api_eigen_analysis():
    res = compute_eigen_analysis()
    return jsonify({"success": True, "result": res})

@app.route('/api/calculus/gradient-descent', methods=['POST'])
def api_gradient_descent():
    data = request.get_json() or {}
    lr = float(data.get('learning_rate', 0.05))
    epochs = int(data.get('epochs', 50))
    res = run_gradient_descent(lr, epochs)
    return jsonify({"success": True, "result": res})

@app.route('/api/calculus/chain-rule', methods=['POST'])
def api_chain_rule():
    data = request.get_json() or {}
    res = compute_chain_rule_step(
        x_val=float(data.get('x', 2.0)),
        target=float(data.get('target', 1.0)),
        w1=float(data.get('w1', 0.5)),
        b1=float(data.get('b1', 0.1)),
        w2=float(data.get('w2', 0.8)),
        b2=float(data.get('b2', -0.2))
    )
    return jsonify({"success": True, "result": res})

if __name__ == '__main__':
    print("=" * 60)
    print("[WEEK 5-6] Math for Machine Learning Explorer")
    print("[SERVER] Starting server at http://127.0.0.1:5002")
    print("=" * 60)
    app.run(debug=True, host='0.0.0.0', port=5002)
