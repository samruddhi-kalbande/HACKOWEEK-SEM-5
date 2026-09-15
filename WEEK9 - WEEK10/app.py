from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import os
from workflow_engine import evaluate_pipeline, run_clustering

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

@app.route('/api/pipeline/evaluate', methods=['GET'])
def api_evaluate():
    res = evaluate_pipeline()
    return jsonify({"success": True, "evaluation": res})

@app.route('/api/clustering/run', methods=['POST'])
def api_clustering():
    data = request.get_json() or {}
    algo = data.get('algorithm', 'kmeans')
    k = int(data.get('k', 3))
    eps = float(data.get('eps', 1.8))
    min_samples = int(data.get('min_samples', 2))

    res = run_clustering(algo, k, eps, min_samples)
    return jsonify({"success": True, "clustering": res})

if __name__ == '__main__':
    print("=" * 60)
    print("[WEEK 9-10] ML Pipeline & Clustering Studio")
    print("[SERVER] Starting server at http://127.0.0.1:5004")
    print("=" * 60)
    app.run(debug=True, host='0.0.0.0', port=5004)
