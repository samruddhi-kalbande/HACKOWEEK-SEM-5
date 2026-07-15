const apiBase = "/api/students";
let studentsCache = [];
let editingStudentId = null;

function showToast(message, type = "info") {
    const container = document.getElementById("toastContainer");
    const toast = document.createElement("div");
    toast.className = `toast ${type}`;
    toast.textContent = message;

    container.appendChild(toast);
    setTimeout(() => {
        toast.classList.add("visible");
    }, 10);

    setTimeout(() => {
        toast.classList.remove("visible");
        setTimeout(() => container.removeChild(toast), 300);
    }, 2800);
}

async function fetchStudents(search = "", course = "") {
    const params = new URLSearchParams();
    if (search) params.set("search", search);
    if (course) params.set("course", course);

    const response = await fetch(`${apiBase}?${params.toString()}`);
    if (!response.ok) {
        throw new Error("Unable to load students");
    }

    return response.json();
}

function setLoadingState(isLoading) {
    const resultsCount = document.getElementById("resultsCount");
    resultsCount.textContent = isLoading ? "Loading students..." : "";
}

function updateDashboard(students) {
    const total = students.length;
    const avgAge = total === 0 ? 0 : (students.reduce((sum, student) => sum + student.age, 0) / total).toFixed(1);
    const uniqueCourses = new Set(students.map(student => student.course.trim())).size;

    document.getElementById("statTotalStudents").textContent = total;
    document.getElementById("statAvgAge").textContent = avgAge;
    document.getElementById("statCourses").textContent = uniqueCourses;
}

function populateCourseFilter(students) {
    const filter = document.getElementById("courseFilter");
    const selectedValue = filter.value;
    const courses = [...new Set(students.map(student => student.course.trim()))].sort();

    filter.innerHTML = `<option value="">All Courses</option>`;
    courses.forEach(course => {
        const option = document.createElement("option");
        option.value = course;
        option.textContent = course;
        filter.appendChild(option);
    });

    if (courses.includes(selectedValue)) {
        filter.value = selectedValue;
    }
}

function renderStudentRow(student) {
    const row = document.createElement("tr");
    row.innerHTML = `
        <td>${student.id}</td>
        <td>${student.name}</td>
        <td>${student.age}</td>
        <td>${student.course}</td>
        <td class="actions-col">
            <button class="btn btn-secondary" onclick="openEditModal(${student.id})">Edit</button>
            <button class="btn btn-danger" onclick="deleteStudent(${student.id})">Delete</button>
        </td>
    `;
    return row;
}

function updateTable(students) {
    const tbody = document.getElementById("studentTableBody");
    const emptyState = document.getElementById("emptyState");

    tbody.innerHTML = "";
    if (students.length === 0) {
        emptyState.style.display = "flex";
        return;
    }

    emptyState.style.display = "none";
    students.forEach(student => tbody.appendChild(renderStudentRow(student)));
}

async function loadStudents() {
    const search = document.getElementById("searchInput").value.trim();
    const course = document.getElementById("courseFilter").value;

    try {
        setLoadingState(true);
        const students = await fetchStudents(search, course);
        studentsCache = students;
        updateTable(students);
        updateDashboard(students);
        populateCourseFilter(studentsCache);
        document.getElementById("resultsCount").textContent = `Showing ${students.length} student${students.length === 1 ? "" : "s"}`;
    } catch (error) {
        showToast(error.message, "error");
    } finally {
        setLoadingState(false);
    }
}

function resetForm() {
    editingStudentId = null;
    document.getElementById("modalTitle").textContent = "Add New Student";
    document.getElementById("submitFormBtn").textContent = "Add Student";
    document.getElementById("actionType").value = "add";
    document.getElementById("studentId").disabled = false;
    document.getElementById("studentId").value = "";
    document.getElementById("studentName").value = "";
    document.getElementById("studentAge").value = "";
    document.getElementById("studentCourse").value = "";
    ["idError", "nameError", "ageError", "courseError"].forEach(id => document.getElementById(id).textContent = "");
}

function openAddModal() {
    resetForm();
    document.getElementById("studentModal").style.display = "flex";
}

function closeModal() {
    document.getElementById("studentModal").style.display = "none";
    resetForm();
}

function openEditModal(studentId) {
    const student = studentsCache.find(s => s.id === studentId);
    if (!student) {
        showToast("Student record not found.", "error");
        return;
    }

    editingStudentId = studentId;
    document.getElementById("modalTitle").textContent = "Edit Student";
    document.getElementById("submitFormBtn").textContent = "Save Changes";
    document.getElementById("actionType").value = "edit";
    document.getElementById("studentId").value = student.id;
    document.getElementById("studentId").disabled = true;
    document.getElementById("studentName").value = student.name;
    document.getElementById("studentAge").value = student.age;
    document.getElementById("studentCourse").value = student.course;
    ["idError", "nameError", "ageError", "courseError"].forEach(id => document.getElementById(id).textContent = "");
    document.getElementById("studentModal").style.display = "flex";
}

function validateStudentForm(id, name, age, course) {
    let isValid = true;
    ["idError", "nameError", "ageError", "courseError"].forEach(id => document.getElementById(id).textContent = "");

    if (!id || id <= 0) {
        document.getElementById("idError").textContent = "Please provide a valid student ID.";
        isValid = false;
    }
    if (!name) {
        document.getElementById("nameError").textContent = "Please enter the student name.";
        isValid = false;
    }
    if (!age || age <= 0 || age > 120) {
        document.getElementById("ageError").textContent = "Please provide a valid age between 1 and 120.";
        isValid = false;
    }
    if (!course) {
        document.getElementById("courseError").textContent = "Please enter the course name.";
        isValid = false;
    }

    return isValid;
}

async function handleFormSubmit(event) {
    event.preventDefault();

    const studentIdInput = document.getElementById("studentId");
    const id = Number(studentIdInput.value.trim());
    const name = document.getElementById("studentName").value.trim();
    const age = Number(document.getElementById("studentAge").value.trim());
    const course = document.getElementById("studentCourse").value.trim();

    if (!validateStudentForm(id, name, age, course)) {
        return;
    }

    const payload = { id, name, age, course };
    const action = document.getElementById("actionType").value;

    try {
        const response = await fetch(action === "edit" ? `${apiBase}/${editingStudentId}` : apiBase, {
            method: action === "edit" ? "PUT" : "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });

        const data = await response.json();
        if (!response.ok) {
            throw new Error(data.error || "Unable to save student record.");
        }

        showToast(data.message, "success");
        closeModal();
        await loadStudents();
    } catch (error) {
        showToast(error.message, "error");
    }
}

async function deleteStudent(studentId) {
    if (!window.confirm("Delete this student record?")) {
        return;
    }

    try {
        const response = await fetch(`${apiBase}/${studentId}`, { method: "DELETE" });
        const data = await response.json();
        if (!response.ok) {
            throw new Error(data.error || "Unable to delete student record.");
        }

        showToast(data.message, "success");
        await loadStudents();
    } catch (error) {
        showToast(error.message, "error");
    }
}

function bindControls() {
    document.getElementById("searchInput").addEventListener("input", () => loadStudents());
    document.getElementById("courseFilter").addEventListener("change", () => loadStudents());
    document.getElementById("studentForm").addEventListener("submit", handleFormSubmit);
}

window.addEventListener("DOMContentLoaded", () => {
    bindControls();
    loadStudents();
});
