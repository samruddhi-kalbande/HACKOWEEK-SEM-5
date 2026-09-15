from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import os
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PUBLIC_DIR = os.path.join(BASE_DIR, 'public')
DATA_DIR = os.path.join(BASE_DIR, 'data')
STUDENTS_CSV = os.path.join(DATA_DIR, 'kaggle_student_performance.csv')
ENROLLMENTS_CSV = os.path.join(DATA_DIR, 'course_enrollments.csv')

app = Flask(__name__, static_folder=PUBLIC_DIR, static_url_path='')
CORS(app)

# ─── Load Data ───────────────────────────────────────────────────────────────
def get_dataframes():
    df_s = pd.read_csv(STUDENTS_CSV)
    df_e = pd.read_csv(ENROLLMENTS_CSV) if os.path.exists(ENROLLMENTS_CSV) else pd.DataFrame()
    return df_s, df_e

# ─── Static Routing ──────────────────────────────────────────────────────────
@app.route('/')
def index():
    return send_from_directory(PUBLIC_DIR, 'index.html')

@app.route('/<path:path>')
def static_proxy(path):
    if os.path.exists(os.path.join(PUBLIC_DIR, path)):
        return send_from_directory(PUBLIC_DIR, path)
    return send_from_directory(PUBLIC_DIR, 'index.html')

# ─── API Endpoints ───────────────────────────────────────────────────────────

@app.route('/api/dataset/summary', methods=['GET'])
def get_dataset_summary():
    df_s, df_e = get_dataframes()
    return jsonify({
        "success": True,
        "students_count": len(df_s),
        "enrollments_count": len(df_e),
        "columns": list(df_s.columns),
        "departments": sorted(df_s['Department'].unique().tolist()),
        "years": sorted(df_s['Year'].unique().tolist()),
        "cgpa_stats": {
            "mean": round(float(df_s['CGPA'].mean()), 2),
            "max": round(float(df_s['CGPA'].max()), 2),
            "min": round(float(df_s['CGPA'].min()), 2),
            "std": round(float(df_s['CGPA'].std()), 2)
        },
        "attendance_stats": {
            "mean": round(float(df_s['AttendanceRate'].mean()), 1),
            "std": round(float(df_s['AttendanceRate'].std()), 1)
        }
    })

@app.route('/api/dataset/records', methods=['GET'])
def get_records():
    df_s, _ = get_dataframes()
    table_type = request.args.get('table', 'students')
    limit = request.args.get('limit', default=25, type=int)

    if table_type == 'enrollments':
        _, df_e = get_dataframes()
        return jsonify({"success": True, "data": df_e.head(limit).to_dict(orient='records')})
    elif table_type == 'merged':
        _, df_e = get_dataframes()
        df_m = pd.merge(df_s, df_e, on='StudentID', how='inner')
        return jsonify({"success": True, "data": df_m.head(limit).to_dict(orient='records')})
    
    return jsonify({"success": True, "data": df_s.head(limit).to_dict(orient='records')})

@app.route('/api/pandas/groupby', methods=['POST'])
def run_groupby():
    df_s, _ = get_dataframes()
    data = request.get_json() or {}

    group_col = data.get('group_by', 'Department')
    metric_col = data.get('metric', 'CGPA')
    agg_func = data.get('agg', 'mean')

    if group_col not in df_s.columns or metric_col not in df_s.columns:
        return jsonify({"success": False, "message": "Invalid columns specified"}), 400

    try:
        grouped = df_s.groupby(group_col).agg(
            Result=(metric_col, agg_func),
            Count=('StudentID', 'count')
        ).round(2).reset_index()

        return jsonify({
            "success": True,
            "group_by": group_col,
            "metric": metric_col,
            "agg": agg_func,
            "records": grouped.to_dict(orient='records')
        })
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

@app.route('/api/numpy/vectorize', methods=['POST'])
def run_vectorize():
    df_s, _ = get_dataframes()
    data = request.get_json() or {}
    
    math_val = float(data.get('math', 80))
    reading_val = float(data.get('reading', 80))
    writing_val = float(data.get('writing', 80))
    
    scores = df_s[['MathScore', 'ReadingScore', 'WritingScore']].to_numpy()
    means = np.mean(scores, axis=0)
    stds = np.std(scores, axis=0)
    
    input_vec = np.array([math_val, reading_val, writing_val])
    input_z = (input_vec - means) / stds
    composite = np.mean(input_vec)
    
    return jsonify({
        "success": True,
        "input_scores": {"Math": math_val, "Reading": reading_val, "Writing": writing_val},
        "subject_means": {"Math": round(float(means[0]), 2), "Reading": round(float(means[1]), 2), "Writing": round(float(means[2]), 2)},
        "computed_z_scores": {"Math": round(float(input_z[0]), 2), "Reading": round(float(input_z[1]), 2), "Writing": round(float(input_z[2]), 2)},
        "composite_score": round(float(composite), 2),
        "percentile_rank": round(float(np.mean(composite >= np.mean(scores, axis=1)) * 100), 1)
    })

@app.route('/api/plots/list', methods=['GET'])
def get_plots():
    return jsonify({
        "success": True,
        "plots": [
            {
                "id": "cgpa_dist",
                "title": "CGPA Distribution & KDE Curve",
                "file": "plots/cgpa_distribution.png",
                "desc": "Kernel Density Estimation and histogram showing normal-like distribution of cumulative grade points."
            },
            {
                "id": "dept_box",
                "title": "Academic Performance by Department",
                "file": "plots/department_boxplot.png",
                "desc": "Box plot with jittered stripplot illustrating medians, quartiles, and variances across branches."
            },
            {
                "id": "study_scatter",
                "title": "Study Hours vs CGPA (OLS Regression)",
                "file": "plots/study_hours_vs_cgpa.png",
                "desc": "Linear regression trendline highlighting strong positive correlation between study time and CGPA."
            },
            {
                "id": "corr_heat",
                "title": "Quantitative Feature Correlation Matrix",
                "file": "plots/correlation_heatmap.png",
                "desc": "Seaborn heatmap of Pearson correlation coefficients across 8 academic & behavioral features."
            }
        ]
    })

if __name__ == '__main__':
    print("=" * 60)
    print("[WEEK 3-4] Data Science Analytics & Visualizer Dashboard")
    print("[SERVER] Starting server at http://127.0.0.1:5001")
    print("=" * 60)
    app.run(debug=True, host='0.0.0.0', port=5001)
