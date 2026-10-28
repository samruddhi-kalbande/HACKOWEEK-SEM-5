import os
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from deep_learning_engine import (
    demonstrate_tensors_and_autograd,
    train_pytorch_loop_demo,
    get_cnn_architecture_data,
    compare_sequence_models,
    compare_regularization_strategies,
    get_transfer_learning_overview,
    analyze_transformers_text,
    query_llm_service
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

@app.route('/api/pytorch/tensors-autograd', methods=['GET'])
def api_tensors_autograd():
    try:
        res = demonstrate_tensors_and_autograd()
        return jsonify({"success": True, "data": res})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/pytorch/training-loop', methods=['POST', 'GET'])
def api_training_loop():
    try:
        if request.method == 'POST':
            payload = request.get_json(force=True) or {}
            epochs = int(payload.get('epochs', 40))
            lr = float(payload.get('learning_rate', 0.01))
        else:
            epochs = int(request.args.get('epochs', 40))
            lr = float(request.args.get('learning_rate', 0.01))

        res = train_pytorch_loop_demo(epochs=epochs, lr=lr)
        return jsonify({"success": True, "data": res})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/cnn/architecture', methods=['GET'])
def api_cnn_architecture():
    try:
        res = get_cnn_architecture_data()
        return jsonify({"success": True, "data": res})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/sequences/compare', methods=['GET'])
def api_sequences():
    try:
        hidden_dim = int(request.args.get('hidden_dim', 32))
        res = compare_sequence_models(hidden_dim=hidden_dim)
        return jsonify({"success": True, "data": res})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/regularization/compare', methods=['GET'])
def api_regularization():
    try:
        res = compare_regularization_strategies()
        return jsonify({"success": True, "data": res})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/transfer-learning/info', methods=['GET'])
def api_transfer_learning():
    try:
        res = get_transfer_learning_overview()
        return jsonify({"success": True, "data": res})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/transformers/analyze', methods=['POST', 'GET'])
def api_transformers():
    try:
        if request.method == 'POST':
            payload = request.get_json(force=True) or {}
            text = payload.get('text', 'Deep learning with PyTorch and Transformers is fascinating!')
        else:
            text = request.args.get('text', 'Deep learning with PyTorch and Transformers is fascinating!')

        res = analyze_transformers_text(input_text=text)
        return jsonify({"success": True, "data": res})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/llm/query', methods=['POST'])
def api_llm_query():
    try:
        payload = request.get_json(force=True) or {}
        prompt = payload.get('prompt', 'Explain PyTorch Autograd and backpropagation.')
        provider = payload.get('provider', 'simulation')
        task_type = payload.get('task_type', 'qa')
        api_key = payload.get('api_key', None)

        res = query_llm_service(prompt=prompt, provider=provider, task_type=task_type, api_key=api_key)
        return jsonify({"success": True, "data": res})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == '__main__':
    print("=" * 65)
    print("[WEEK 17-18] PyTorch, CNNs, RNNs, Regularization, HF & LLMs Studio")
    print("[SERVER] Running at: http://127.0.0.1:5008")
    print("=" * 65)
    app.run(debug=True, host='0.0.0.0', port=5008)
