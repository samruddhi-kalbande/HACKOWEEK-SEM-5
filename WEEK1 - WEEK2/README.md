# Week 1 – Week 2: Student Information Management System (SIMS) REST API & Web Application

[![REST API](https://img.shields.io/badge/API-RESTful-blue.svg)](https://restfulapi.net/)
[![Flask](https://img.shields.io/badge/Backend-Flask%203.1-green.svg)](https://flask.palletsprojects.com/)
[![Dataset](https://img.shields.io/badge/Dataset-Kaggle%20Academic%20Insights-orange.svg)](https://www.kaggle.com/)
[![Activity](https://img.shields.io/badge/Activity-Whiteboard%20Challenge-purple.svg)](#whiteboard-challenge-activity)

## 📌 Project Overview
Developed for **HACKOWEEK SEM 5 (Third Year Undergraduate Program)** covering **Week 1 – Week 2**:
- **Topic / Skill**: APIs & Backend Basics (Flask / Node.js)
- **Mini-Project**: Build a REST API for a **Student Information Management System (SIMS)**
- **Dataset**: Curated **Kaggle Student Performance & Academic Insights Dataset** in CSV and JSON formats, designed as the persistent benchmark for all subsequent weeks (Weeks 1 to 10).
- **Web Application**: Interactive, responsive frontend dashboard featuring real-time analytics, filtering, CRUD modals, an interactive API Explorer console, and Whiteboard Challenge documentation.
- **Activity**: Comprehensive Whiteboard Challenge material covering REST architectural constraints, HTTP status codes, and system design.

---

## 🚀 Quickstart Guide

### 1. Requirements
- Python 3.9+ installed
- `flask` and `flask-cors`

```bash
pip install flask flask-cors
```

### 2. Start the Application
Run the Flask server from the project directory:

```bash
cd "WEEK1 - WEEK2/WEEK1 - WEEK2"
python app.py
```

### 3. Access the Web Dashboard
Open your browser and navigate to:
```
http://localhost:5000
```

---

## 📡 REST API Endpoints Specification

| Method | Endpoint | Description | Sample Query / Body |
|---|---|---|---|
| `GET` | `/api/students` | Get all students (search, filter, sort) | `?branch=Computer Science&year=3&min_cgpa=8.5` |
| `GET` | `/api/students/<id>` | Get student profile by ID | Path parameter: `/api/students/STU001` |
| `POST` | `/api/students` | Add a new student record | JSON: `{"name":"...","email":"...","branch":"...","year":3,"cgpa":8.9}` |
| `PUT` | `/api/students/<id>` | Update existing student record | JSON: `{"cgpa": 9.20, "attendance": 95}` |
| `DELETE` | `/api/students/<id>` | Remove student record | Path parameter: `/api/students/STU020` |
| `GET` | `/api/stats` | Dashboard analytical summary | Aggregate metrics, department and CGPA distribution |
| `GET` | `/api/dataset-info` | Kaggle dataset metadata | Information card and 10-week roadmap |

---

## 📊 Dataset Attribution & Roadmap
- Raw CSV: `data/kaggle_student_performance.csv`
- Active API Store: `data/students.json`
- Feature specifications & 10-week roadmap: [DATASET_INFO.md](data/DATASET_INFO.md)

---

## 📝 Whiteboard Challenge Activity
- Complete reference notes, diagrams, and common interview questions on REST architecture, idempotency, and status codes are integrated directly into the web application's **"Whiteboard Challenge"** tab.
