from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import json
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PUBLIC_DIR = os.path.join(BASE_DIR, 'public')
DATA_FILE = os.path.join(BASE_DIR, 'data', 'students.json')
CSV_FILE = os.path.join(BASE_DIR, 'data', 'kaggle_student_performance.csv')

app = Flask(__name__, static_folder=PUBLIC_DIR, static_url_path='')
CORS(app)

# ─── Helper Functions ────────────────────────────────────────────────────────

def load_students():
    if not os.path.exists(DATA_FILE):
        return []
    with open(DATA_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_students(students):
    with open(DATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(students, f, indent=2, ensure_ascii=False)

# ─── Serve Frontend ──────────────────────────────────────────────────────────

@app.route('/')
def index():
    return send_from_directory(PUBLIC_DIR, 'index.html')

@app.route('/<path:path>')
def static_proxy(path):
    if os.path.exists(os.path.join(PUBLIC_DIR, path)):
        return send_from_directory(PUBLIC_DIR, path)
    return send_from_directory(PUBLIC_DIR, 'index.html')

# ─── API Routes ──────────────────────────────────────────────────────────────

# GET /api/students - List students with search, filters, sorting & pagination
@app.route('/api/students', methods=['GET'])
def get_students():
    students = load_students()

    # Query parameters
    year     = request.args.get('year')
    branch   = request.args.get('branch')
    status   = request.args.get('status')
    search   = request.args.get('search', '').strip().lower()
    min_cgpa = request.args.get('min_cgpa', type=float)
    max_cgpa = request.args.get('max_cgpa', type=float)
    sort_by  = request.args.get('sort_by', 'id')
    order    = request.args.get('order', 'asc').lower()

    # Filters
    if year:
        students = [s for s in students if str(s.get('year')) == str(year)]
    if branch:
        students = [s for s in students if s.get('branch', '').lower() == branch.lower()]
    if status:
        students = [s for s in students if s.get('status', '').lower() == status.lower()]
    if min_cgpa is not None:
        students = [s for s in students if float(s.get('cgpa', 0)) >= min_cgpa]
    if max_cgpa is not None:
        students = [s for s in students if float(s.get('cgpa', 0)) <= max_cgpa]
    if search:
        students = [
            s for s in students if
            search in str(s.get('name', '')).lower() or
            search in str(s.get('email', '')).lower() or
            search in str(s.get('id', '')).lower() or
            search in str(s.get('city', '')).lower() or
            any(search in str(skill).lower() for skill in s.get('skills', []))
        ]

    # Sorting
    reverse = (order == 'desc')
    if sort_by in ['cgpa', 'attendance', 'age', 'year', 'study_hours_weekly', 'math_score']:
        students.sort(key=lambda s: float(s.get(sort_by, 0)), reverse=reverse)
    elif sort_by in ['name', 'id', 'branch', 'city']:
        students.sort(key=lambda s: str(s.get(sort_by, '')).lower(), reverse=reverse)

    total_matched = len(students)

    # Optional pagination
    page = request.args.get('page', type=int)
    limit = request.args.get('limit', type=int)
    if page and limit:
        start = (page - 1) * limit
        end = start + limit
        paginated_students = students[start:end]
    else:
        paginated_students = students

    return jsonify({
        "success": True,
        "count": len(paginated_students),
        "total": total_matched,
        "students": paginated_students
    })

# GET /api/students/<student_id> - Retrieve single student
@app.route('/api/students/<student_id>', methods=['GET'])
def get_student(student_id):
    students = load_students()
    student = next((s for s in students if s.get('id') == student_id), None)
    if not student:
        return jsonify({"success": False, "message": f"Student with ID '{student_id}' not found"}), 404
    return jsonify({"success": True, "student": student})

# POST /api/students - Create new student
@app.route('/api/students', methods=['POST'])
def add_student():
    students = load_students()
    data = request.get_json() or {}

    required = ['name', 'email', 'branch', 'year', 'cgpa']
    missing = [f for f in required if f not in data or data[f] is None or str(data[f]).strip() == '']
    if missing:
        return jsonify({
            "success": False,
            "message": f"Missing required fields: {', '.join(missing)}"
        }), 400

    # Validate duplicate email
    if any(s.get('email', '').lower() == str(data['email']).strip().lower() for s in students):
        return jsonify({
            "success": False,
            "message": f"Student with email '{data['email']}' already exists."
        }), 409

    # Generate sequential ID if not specified
    if 'id' in data and data['id']:
        new_id = str(data['id']).strip().upper()
        if any(s.get('id') == new_id for s in students):
            return jsonify({"success": False, "message": f"Student ID '{new_id}' already exists."}), 409
    else:
        existing_numeric_ids = []
        for s in students:
            id_val = str(s.get('id', ''))
            digits = ''.join(c for c in id_val if c.isdigit())
            if digits:
                existing_numeric_ids.append(int(digits))
        next_num = (max(existing_numeric_ids) + 1) if existing_numeric_ids else 1
        new_id = f"STU{str(next_num).zfill(3)}"

    # Format fields and apply defaults
    new_student = {
        "id": new_id,
        "name": str(data['name']).strip(),
        "gender": str(data.get('gender', 'Male')).strip(),
        "age": int(data.get('age', 20)),
        "city": str(data.get('city', 'Unknown')).strip(),
        "email": str(data['email']).strip(),
        "phone": str(data.get('phone', '+91-0000000000')).strip(),
        "branch": str(data['branch']).strip(),
        "year": int(data['year']),
        "semester": int(data.get('semester', int(data['year']) * 2)),
        "cgpa": round(float(data['cgpa']), 2),
        "attendance": round(float(data.get('attendance', 80.0)), 1),
        "study_hours_weekly": round(float(data.get('study_hours_weekly', 12.0)), 1),
        "math_score": int(data.get('math_score', 75)),
        "reading_score": int(data.get('reading_score', 75)),
        "writing_score": int(data.get('writing_score', 75)),
        "skills": data.get('skills', []) if isinstance(data.get('skills'), list) else [s.strip() for s in str(data.get('skills', '')).split(',') if s.strip()],
        "projects": data.get('projects', []) if isinstance(data.get('projects'), list) else [p.strip() for p in str(data.get('projects', '')).split(',') if p.strip()],
        "activities": data.get('activities', []) if isinstance(data.get('activities'), list) else [a.strip() for a in str(data.get('activities', '')).split(',') if a.strip()],
        "status": str(data.get('status', 'Active')).capitalize()
    }

    students.append(new_student)
    save_students(students)

    return jsonify({
        "success": True,
        "message": f"Student '{new_student['name']}' created successfully.",
        "student": new_student
    }), 201

# PUT /api/students/<student_id> - Update existing student
@app.route('/api/students/<student_id>', methods=['PUT'])
def update_student(student_id):
    students = load_students()
    idx = next((i for i, s in enumerate(students) if s.get('id') == student_id), None)
    if idx is None:
        return jsonify({"success": False, "message": f"Student with ID '{student_id}' not found"}), 404

    updates = request.get_json() or {}
    updates.pop('id', None)  # Prevent mutating ID

    # Type conversions if updating numeric fields
    if 'cgpa' in updates:
        updates['cgpa'] = round(float(updates['cgpa']), 2)
    if 'attendance' in updates:
        updates['attendance'] = round(float(updates['attendance']), 1)
    if 'year' in updates:
        updates['year'] = int(updates['year'])
    if 'age' in updates:
        updates['age'] = int(updates['age'])
    if 'skills' in updates and isinstance(updates['skills'], str):
        updates['skills'] = [s.strip() for s in updates['skills'].split(',') if s.strip()]
    if 'projects' in updates and isinstance(updates['projects'], str):
        updates['projects'] = [p.strip() for p in updates['projects'].split(',') if p.strip()]
    if 'activities' in updates and isinstance(updates['activities'], str):
        updates['activities'] = [a.strip() for a in updates['activities'].split(',') if a.strip()]

    students[idx].update(updates)
    save_students(students)

    return jsonify({
        "success": True,
        "message": f"Student '{student_id}' updated successfully.",
        "student": students[idx]
    })

# DELETE /api/students/<student_id> - Remove student
@app.route('/api/students/<student_id>', methods=['DELETE'])
def delete_student(student_id):
    students = load_students()
    initial_count = len(students)
    students = [s for s in students if s.get('id') != student_id]

    if len(students) == initial_count:
        return jsonify({"success": False, "message": f"Student with ID '{student_id}' not found"}), 404

    save_students(students)
    return jsonify({
        "success": True,
        "message": f"Student '{student_id}' has been deleted successfully."
    })

# GET /api/stats - Analytical summary for the dashboard
@app.route('/api/stats', methods=['GET'])
def get_stats():
    students = load_students()
    total = len(students)
    if total == 0:
        return jsonify({"success": True, "stats": {}})

    active_count = sum(1 for s in students if s.get('status', '').lower() == 'active')
    avg_cgpa = round(sum(s.get('cgpa', 0) for s in students) / total, 2)
    avg_attendance = round(sum(s.get('attendance', 0) for s in students) / total, 1)
    avg_study_hours = round(sum(s.get('study_hours_weekly', 0) for s in students) / total, 1)

    top_student = max(students, key=lambda s: s.get('cgpa', 0))

    # Branch counts
    branches = {}
    for s in students:
        b = s.get('branch', 'Other')
        branches[b] = branches.get(b, 0) + 1

    # Year counts
    years = {}
    for s in students:
        y = f"Year {s.get('year', '?')}"
        years[y] = years.get(y, 0) + 1

    # CGPA distributions
    cgpa_ranges = {
        "Below 7.0": sum(1 for s in students if s.get('cgpa', 0) < 7.0),
        "7.0 - 7.9": sum(1 for s in students if 7.0 <= s.get('cgpa', 0) < 8.0),
        "8.0 - 8.9": sum(1 for s in students if 8.0 <= s.get('cgpa', 0) < 9.0),
        "9.0 and Above": sum(1 for s in students if s.get('cgpa', 0) >= 9.0)
    }

    # Gender breakdown
    gender_counts = {}
    for s in students:
        g = s.get('gender', 'Other')
        gender_counts[g] = gender_counts.get(g, 0) + 1

    return jsonify({
        "success": True,
        "stats": {
            "total_students": total,
            "active_students": active_count,
            "inactive_students": total - active_count,
            "average_cgpa": avg_cgpa,
            "average_attendance": avg_attendance,
            "average_study_hours": avg_study_hours,
            "top_student": {
                "id": top_student.get('id'),
                "name": top_student.get('name'),
                "branch": top_student.get('branch'),
                "cgpa": top_student.get('cgpa')
            },
            "branch_distribution": branches,
            "year_distribution": years,
            "cgpa_distribution": cgpa_ranges,
            "gender_distribution": gender_counts
        }
    })

# GET /api/dataset-info - Kaggle dataset metadata
@app.route('/api/dataset-info', methods=['GET'])
def get_dataset_info():
    students = load_students()
    return jsonify({
        "success": True,
        "dataset": {
            "name": "Kaggle Student Performance & Academic Insights Dataset",
            "format": "CSV & JSON",
            "records_count": len(students),
            "csv_file": "data/kaggle_student_performance.csv",
            "json_file": "data/students.json",
            "primary_features": [
                "StudentID", "Name", "Department/Branch", "Year", "Semester",
                "CGPA", "AttendanceRate", "StudyHoursPerWeek",
                "MathScore", "ReadingScore", "WritingScore",
                "Skills", "Projects", "Activities", "Status"
            ],
            "multi_week_roadmap": [
                "Week 1-2: REST API & Web Dashboard",
                "Week 3-4: Data Structures, Sorting & Indexing Algorithms",
                "Week 5-6: Relational Database & ORM Integration",
                "Week 7-8: Exploratory Data Analysis & Visual Analytics",
                "Week 9-10: Predictive ML Modeling & Performance Classification"
            ]
        }
    })

# ─── Main ────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 60)
    print("[SIMS] Student Information Management System")
    print("[SERVER] REST API Server starting at http://127.0.0.1:5000")
    print("[PUBLIC] Public directory: " + str(PUBLIC_DIR))
    print("[DATA] Active Dataset: " + str(DATA_FILE))
    print("=" * 60)
    app.run(debug=True, host='0.0.0.0', port=5000)

