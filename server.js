const express = require("express");
const fs = require("fs").promises;
const path = require("path");

const app = express();
app.use(express.json());
app.use(express.static(path.join(__dirname, "public")));

const PORT = process.env.PORT || 5000;
const DATA_FILE = path.join(__dirname, "data", "students.json");

async function readStudents() {
    try {
        const data = await fs.readFile(DATA_FILE, "utf8");
        return JSON.parse(data);
    } catch (error) {
        if (error.code === "ENOENT") {
            await fs.mkdir(path.dirname(DATA_FILE), { recursive: true });
            await fs.writeFile(DATA_FILE, "[]", "utf8");
            return [];
        }
        console.error("Error reading students file:", error);
        throw error;
    }
}

async function writeStudents(students) {
    try {
        await fs.writeFile(DATA_FILE, JSON.stringify(students, null, 2), "utf8");
    } catch (error) {
        console.error("Error writing students file:", error);
        throw error;
    }
}

app.get("/api/students", async (req, res) => {
    try {
        let students = await readStudents();
        const { search, course } = req.query;

        if (search) {
            const query = search.toLowerCase();
            students = students.filter(s =>
                s.name.toLowerCase().includes(query) ||
                String(s.id).includes(query) ||
                s.course.toLowerCase().includes(query)
            );
        }

        if (course) {
            students = students.filter(s => s.course.toLowerCase() === course.toLowerCase());
        }

        res.json(students);
    } catch (error) {
        res.status(500).json({ error: "Failed to fetch students" });
    }
});

app.get("/api/students/:id", async (req, res) => {
    try {
        const students = await readStudents();
        const id = Number(req.params.id);
        const student = students.find(s => s.id === id);

        if (!student) {
            return res.status(404).json({ error: "Student not found" });
        }

        res.json(student);
    } catch (error) {
        res.status(500).json({ error: "Failed to fetch student" });
    }
});

app.post("/api/students", async (req, res) => {
    try {
        const students = await readStudents();
        const { id, name, age, course } = req.body;

        if (!id || !name || !age || !course) {
            return res.status(400).json({ error: "All fields (id, name, age, course) are required" });
        }

        const studentId = Number(id);
        const studentAge = Number(age);

        if (isNaN(studentId) || studentId <= 0) {
            return res.status(400).json({ error: "Student ID must be a positive number" });
        }

        if (isNaN(studentAge) || studentAge <= 0 || studentAge > 120) {
            return res.status(400).json({ error: "Age must be a valid positive number" });
        }

        if (students.some(s => s.id === studentId)) {
            return res.status(409).json({ error: `Student with ID ${studentId} already exists` });
        }

        const newStudent = {
            id: studentId,
            name: name.trim(),
            age: studentAge,
            course: course.trim()
        };

        students.push(newStudent);
        await writeStudents(students);

        res.status(201).json({ message: "Student added successfully", student: newStudent });
    } catch (error) {
        res.status(500).json({ error: "Failed to add student" });
    }
});

app.put("/api/students/:id", async (req, res) => {
    try {
        const students = await readStudents();
        const id = Number(req.params.id);
        const studentIndex = students.findIndex(s => s.id === id);

        if (studentIndex === -1) {
            return res.status(404).json({ error: "Student not found" });
        }

        const { name, age, course } = req.body;
        if (!name || !age || !course) {
            return res.status(400).json({ error: "All fields (name, age, course) are required for update" });
        }

        const studentAge = Number(age);
        if (isNaN(studentAge) || studentAge <= 0 || studentAge > 120) {
            return res.status(400).json({ error: "Age must be a valid positive number" });
        }

        students[studentIndex] = {
            id,
            name: name.trim(),
            age: studentAge,
            course: course.trim()
        };

        await writeStudents(students);

        res.json({ message: "Student updated successfully", student: students[studentIndex] });
    } catch (error) {
        res.status(500).json({ error: "Failed to update student" });
    }
});

app.delete("/api/students/:id", async (req, res) => {
    try {
        const students = await readStudents();
        const id = Number(req.params.id);

        if (!students.some(s => s.id === id)) {
            return res.status(404).json({ error: "Student not found" });
        }

        const updatedStudents = students.filter(s => s.id !== id);
        await writeStudents(updatedStudents);

        res.json({ message: `Student with ID ${id} deleted successfully` });
    } catch (error) {
        res.status(500).json({ error: "Failed to delete student" });
    }
});

app.listen(PORT, () => {
    console.log(`Server running on http://localhost:${PORT}`);
});
