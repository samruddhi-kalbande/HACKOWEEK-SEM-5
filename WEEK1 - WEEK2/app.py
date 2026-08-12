from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import or_
from datetime import datetime

app = Flask(__name__, static_folder='public', static_url_path='')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///students.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

class Student(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), nullable=False)
    age = db.Column(db.Integer, nullable=False)
    course = db.Column(db.String(120), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'age': self.age,
            'course': self.course,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

def create_tables():
    with app.app_context():
        db.create_all()

create_tables()

@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.route('/api/students', methods=['GET'])
@app.route('/students', methods=['GET'])
def list_students():
    search = request.args.get('search', '').strip()
    course = request.args.get('course', '').strip()

    query = Student.query
    if search:
        like = f"%{search}%"
        filters = [
            Student.name.ilike(like),
            Student.course.ilike(like)
        ]
        if search.isdigit():
            filters.append(Student.id == int(search))
        query = query.filter(or_(*filters))

    if course:
        query = query.filter(Student.course.ilike(course))

    students = [s.to_dict() for s in query.order_by(Student.id).all()]
    return jsonify(students)

@app.route('/api/students/<int:student_id>', methods=['GET'])
@app.route('/students/<int:student_id>', methods=['GET'])
def get_student(student_id):
    student = Student.query.get_or_404(student_id)
    return jsonify(student.to_dict())

@app.route('/api/students', methods=['POST'])
@app.route('/students', methods=['POST'])
def create_student():
    data = request.get_json() or {}
    required = ['id', 'name', 'age', 'course']
    for field in required:
        if field not in data:
            return jsonify({'error': f'{field} is required'}), 400

    try:
        student_id = int(data['id'])
    except (ValueError, TypeError):
        return jsonify({'error': 'id must be a number'}), 400

    if student_id <= 0:
        return jsonify({'error': 'id must be a positive number'}), 400

    try:
        student_age = int(data['age'])
    except (ValueError, TypeError):
        return jsonify({'error': 'age must be a number'}), 400

    if student_age <= 0 or student_age > 120:
        return jsonify({'error': 'age must be between 1 and 120'}), 400

    name = data['name'].strip()
    course = data['course'].strip()
    if not name or not course:
        return jsonify({'error': 'name and course are required'}), 400

    if Student.query.get(student_id):
        return jsonify({'error': f'Student with id {student_id} already exists'}), 409

    student = Student(id=student_id, name=name, age=student_age, course=course)
    db.session.add(student)
    db.session.commit()
    return jsonify(student.to_dict()), 201

@app.route('/api/students/<int:student_id>', methods=['PUT'])
@app.route('/students/<int:student_id>', methods=['PUT'])
def update_student(student_id):
    student = Student.query.get_or_404(student_id)
    data = request.get_json() or {}

    if 'name' in data:
        name = data.get('name', '').strip()
        if not name:
            return jsonify({'error': 'name is required'}), 400
        student.name = name

    if 'age' in data:
        try:
            student_age = int(data['age'])
        except (ValueError, TypeError):
            return jsonify({'error': 'age must be a number'}), 400
        if student_age <= 0 or student_age > 120:
            return jsonify({'error': 'age must be between 1 and 120'}), 400
        student.age = student_age

    if 'course' in data:
        course = data.get('course', '').strip()
        if not course:
            return jsonify({'error': 'course is required'}), 400
        student.course = course

    db.session.commit()
    return jsonify(student.to_dict())

@app.route('/api/students/<int:student_id>', methods=['DELETE'])
@app.route('/students/<int:student_id>', methods=['DELETE'])
def delete_student(student_id):
    student = Student.query.get_or_404(student_id)
    db.session.delete(student)
    db.session.commit()
    return jsonify({'message': f'Student with id {student_id} deleted successfully'})

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
