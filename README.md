# HACKOWEEK SEM 5

Student Information Management System (SIMS) built with Node.js and Express. This project includes a REST API backend for student CRUD operations and a responsive frontend dashboard to manage student records.

## Features

- Create, read, update, and delete student records
- Search and course filter support
- JSON file-backed storage for simplicity
- Responsive dashboard UI with interactive table and modal forms
- Clean REST API endpoints for integration with other applications

## Tech Stack

- Node.js
- Express
- HTML/CSS/JavaScript frontend

## Getting Started

### Prerequisites

- Node.js 18+ installed

### Install Dependencies

```bash
npm install
```

### Run the Application

```bash
npm start
```

Then open `http://localhost:5000` in your browser.

## API Documentation

### Base URL

`http://localhost:5000/api`

### Endpoints

- `GET /api/students` - Get all students. Optional query parameters: `search`, `course`
- `GET /api/students/:id` - Get a specific student by ID
- `POST /api/students` - Create a new student
  - Request body: `{ id, name, age, course }`
- `PUT /api/students/:id` - Update an existing student
  - Request body: `{ name, age, course }`
- `DELETE /api/students/:id` - Delete a student

## Example Requests

```bash
curl http://localhost:5000/api/students
curl -X POST http://localhost:5000/api/students -H "Content-Type: application/json" -d '{"id":3,"name":"Anita","age":19,"course":"Physics"}'
```

## GitHub Repository

This project is linked to a GitHub repository and can be pushed using the `gh` CLI.
