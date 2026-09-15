# Kaggle Student Performance & Academic Insights Dataset

## 📌 Dataset Overview
This dataset originates from standard Kaggle educational benchmarks (e.g. *Higher Education Students Performance Evaluation* and *Student Performance & Academic Insights*). It contains holistic academic, demographic, behavioural, and co-curricular metrics for engineering and technology undergraduate students.

This dataset will serve as the **foundational dataset across all weeks (Week 1 through Week 10)** in HACKOWEEK SEM 5.

---

## 📁 File Structure
- **CSV Format**: `kaggle_student_performance.csv`
- **JSON REST API Format**: `students.json`

---

## 📊 Feature Dictionary

| Column Name | Type | Description | Example Values |
|---|---|---|---|
| `StudentID` (`id`) | String | Unique student identifier | `STU001`, `STU002` |
| `Name` (`name`) | String | Full name of the student | Aarav Sharma, Priya Deshmukh |
| `Gender` (`gender`) | String | Gender category | `Male`, `Female` |
| `Age` (`age`) | Integer | Student age in years | `19` - `23` |
| `City` (`city`) | String | Origin / current residence city | `Pune`, `Mumbai`, `Bengaluru` |
| `Email` (`email`) | String | Official academic email | `aarav.sharma@college.edu` |
| `Phone` (`phone`) | String | Contact phone number | `+91-9876543210` |
| `Department` (`branch`) | String | Academic engineering branch | `Computer Science`, `IT`, `Data Science`, `Electronics` |
| `Year` (`year`) | Integer | Current study year (1 to 4) | `1`, `2`, `3`, `4` |
| `Semester` (`semester`) | Integer | Current semester (1 to 8) | `1` to `8` |
| `CGPA` (`cgpa`) | Float | Cumulative Grade Point Average (0-10 scale) | `9.35`, `8.72`, `7.85` |
| `AttendanceRate` (`attendance`) | Float | Class & lab attendance percentage | `72.0%` to `97.0%` |
| `StudyHoursPerWeek` (`study_hours_weekly`) | Float | Independent weekly study hours | `8.0` to `25.0` hrs |
| `MathScore` (`math_score`) | Integer | Standardized math/quantitative assessment (out of 100) | `68` to `99` |
| `ReadingScore` (`reading_score`) | Integer | Reading & comprehension score (out of 100) | `71` to `96` |
| `WritingScore` (`writing_score`) | Integer | Technical writing & documentation score (out of 100) | `67` to `97` |
| `Skills` (`skills`) | Array/Delimited | Technical skills & competencies | `Python`, `Flask`, `React`, `Docker` |
| `Projects` (`projects`) | Array/Delimited | Capstone / mini-projects completed | `Sentiment Analyzer`, `Student Portal` |
| `Activities` (`activities`) | Array/Delimited | Extracurriculars, hackathons, clubs | `Whiteboard Challenge`, `Hackathon Lead` |
| `Status` (`status`) | String | Academic standing status | `Active`, `Inactive` |

---

## 🗺️ Multi-Week Utilization Roadmap (Weeks 1 - 10)

- **Week 1 – Week 2 (Current)**:
  - **Topic**: APIs & Backend Basics (Flask / Node.js)
  - **Task**: Implement REST API (`GET`, `POST`, `PUT`, `DELETE`, `/api/stats`) and interactive Student Information Management System frontend.
  - **Activity**: Whiteboard Challenge on REST architectural constraints and system design.
- **Week 3 – Week 4**:
  - **Data Structures & Sorting**: Searching by ID/name, sorting students by CGPA/attendance using QuickSort / MergeSort, BST index on student IDs.
- **Week 5 – Week 6**:
  - **Database Migration & ORM**: Migrate CSV/JSON storage to SQLite / PostgreSQL, index relations between students, courses, and marks.
- **Week 7 – Week 8**:
  - **Exploratory Data Analysis (EDA) & Visualization**: Correlation heatmaps (Study Hours vs CGPA, Attendance vs Exam Scores), branch performance distributions.
- **Week 9 – Week 10**:
  - **Machine Learning / Predictive Modeling**: Predict student CGPA / academic risk (Pass/Fail or Dropout risk) based on attendance, study hours, and exam scores.
