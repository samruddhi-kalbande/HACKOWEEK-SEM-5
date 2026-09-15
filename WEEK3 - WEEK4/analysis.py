import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server-side generation
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
STUDENTS_CSV = os.path.join(DATA_DIR, 'kaggle_student_performance.csv')
ENROLLMENTS_CSV = os.path.join(DATA_DIR, 'course_enrollments.csv')
PLOTS_DIR = os.path.join(BASE_DIR, 'public', 'plots')
os.makedirs(PLOTS_DIR, exist_ok=True)

# Set global seaborn styling
sns.set_theme(style="darkgrid", palette="muted")
plt.rcParams.update({
    'figure.facecolor': '#0e1320',
    'axes.facecolor': '#151c2e',
    'axes.edgecolor': '#2a3650',
    'text.color': '#f1f5f9',
    'axes.labelcolor': '#94a3b8',
    'xtick.color': '#94a3b8',
    'ytick.color': '#94a3b8',
    'grid.color': '#202b44',
    'font.sans-serif': 'Segoe UI',
})

# ─── 1. Python Essentials: OOP & Comprehensions ─────────────────────────────

class StudentRecord:
    """Demonstrates Object-Oriented Programming (OOP) in Python."""
    def __init__(self, student_id, name, department, year, cgpa, attendance, skills):
        self.student_id = student_id
        self.name = name
        self.department = department
        self.year = int(year)
        self._cgpa = float(cgpa)  # Encapsulation
        self.attendance = float(attendance)
        self.skills = skills if isinstance(skills, list) else [s.strip() for s in str(skills).split(';') if s.strip()]

    @property
    def cgpa(self):
        return self._cgpa

    @property
    def academic_standing(self):
        if self._cgpa >= 9.0:
            return "First Class with Distinction"
        elif self._cgpa >= 8.0:
            return "First Class"
        elif self._cgpa >= 7.0:
            return "Second Class"
        else:
            return "Pass"

    def is_eligible_for_placement(self, min_cgpa=8.0, min_attendance=85.0):
        return self._cgpa >= min_cgpa and self.attendance >= min_attendance

    def to_dict(self):
        return {
            "id": self.student_id,
            "name": self.name,
            "department": self.department,
            "year": self.year,
            "cgpa": self._cgpa,
            "standing": self.academic_standing,
            "eligible": self.is_eligible_for_placement(),
            "skills_count": len(self.skills)
        }

def run_python_essentials_demo():
    df = pd.read_csv(STUDENTS_CSV)
    
    # List comprehension: Instantiate OOP objects
    students = [
        StudentRecord(row['StudentID'], row['Name'], row['Department'], row['Year'], row['CGPA'], row['AttendanceRate'], row['Skills'])
        for _, row in df.iterrows()
    ]
    
    # Dict comprehension: Placement eligibility mapping
    placement_map = {s.student_id: s.is_eligible_for_placement() for s in students}
    
    # Filtering with list comprehension
    distinction_students = [s.name for s in students if s.academic_standing == "First Class with Distinction"]
    
    return {
        "total_oop_records": len(students),
        "distinction_count": len(distinction_students),
        "distinction_students": distinction_students,
        "eligible_for_placement_count": sum(1 for v in placement_map.values() if v),
        "sample_oop_summary": [s.to_dict() for s in students[:5]]
    }

# ─── 2. NumPy: Arrays, Broadcasting & Vectorized Ops ────────────────────────

def run_numpy_demo():
    df = pd.read_csv(STUDENTS_CSV)
    
    # Create 2D NumPy array of standardized exam scores: [Math, Reading, Writing]
    scores_matrix = df[['MathScore', 'ReadingScore', 'WritingScore']].to_numpy(dtype=float)
    
    # 1. Vectorized row-wise calculations (Mean & Composite Exam Score)
    composite_scores = np.mean(scores_matrix, axis=1)
    
    # 2. Vectorized column-wise calculations (Mean & Std Dev per subject)
    col_means = np.mean(scores_matrix, axis=0)
    col_stds = np.std(scores_matrix, axis=0)
    
    # 3. Broadcasting: Z-Score Standardization -> (X - mu) / sigma
    # (20, 3) broadcasted against (3,)
    z_scores = (scores_matrix - col_means) / col_stds
    
    # 4. Vectorized CGPA scaling (Converting 0-10 CGPA to 0-100 scale using vectorized multiplication)
    cgpa_vec = df['CGPA'].to_numpy(dtype=float)
    cgpa_percentage = cgpa_vec * 9.5  # Standard AICTE formula
    
    # 5. Boolean masking with NumPy
    top_performers_mask = (composite_scores >= 90.0) & (cgpa_vec >= 9.0)
    top_student_ids = df['StudentID'].to_numpy()[top_performers_mask].tolist()
    
    return {
        "scores_matrix_shape": list(scores_matrix.shape),
        "subject_means": {"Math": round(col_means[0], 2), "Reading": round(col_means[1], 2), "Writing": round(col_means[2], 2)},
        "subject_stds": {"Math": round(col_stds[0], 2), "Reading": round(col_stds[1], 2), "Writing": round(col_stds[2], 2)},
        "broadcasting_z_score_sample": np.round(z_scores[:3], 2).tolist(),
        "top_performers_count": int(np.sum(top_performers_mask)),
        "top_performers_ids": top_student_ids
    }

# ─── 3. Pandas: DataFrames, Cleaning, Merging & GroupBy ──────────────────────

def run_pandas_demo():
    # Load datasets
    df_students = pd.read_csv(STUDENTS_CSV)
    df_enrollments = pd.read_csv(ENROLLMENTS_CSV)
    
    # Data Cleaning: checking nulls, type conversions
    df_students['Age'] = pd.to_numeric(df_students['Age'], errors='coerce')
    df_students['AttendanceRate'] = df_students['AttendanceRate'].clip(upper=100.0)
    
    # Merge operation: INNER JOIN on StudentID
    df_merged = pd.merge(df_students, df_enrollments, on='StudentID', how='inner')
    
    # GroupBy 1: Department wise performance
    dept_summary = df_students.groupby('Department').agg(
        student_count=('StudentID', 'count'),
        avg_cgpa=('CGPA', 'mean'),
        avg_attendance=('AttendanceRate', 'mean'),
        avg_study_hours=('StudyHoursPerWeek', 'mean')
    ).round(2).reset_index()
    
    # GroupBy 2: Year wise academic statistics
    year_summary = df_students.groupby('Year').agg(
        count=('StudentID', 'count'),
        avg_cgpa=('CGPA', 'mean'),
        max_cgpa=('CGPA', 'max'),
        min_cgpa=('CGPA', 'min')
    ).round(2).reset_index()
    
    # GroupBy 3: Course wise average grade points from merged dataframe
    course_summary = df_merged.groupby(['CourseCode', 'CourseName']).agg(
        enrolled=('StudentID', 'count'),
        avg_gradepoints=('GradePoints', 'mean')
    ).round(2).reset_index()
    
    return {
        "raw_students_count": len(df_students),
        "raw_enrollments_count": len(df_enrollments),
        "merged_rows_count": len(df_merged),
        "department_summary": dept_summary.to_dict(orient='records'),
        "year_summary": year_summary.to_dict(orient='records'),
        "course_summary": course_summary.to_dict(orient='records')
    }

# ─── 4. Data Visualization: Matplotlib & Seaborn Plots ──────────────────────

def generate_all_plots():
    df = pd.read_csv(STUDENTS_CSV)
    generated = {}
    
    # Plot 1: CGPA Distribution & KDE
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.histplot(df['CGPA'], kde=True, color='#6366f1', ax=ax, bins=8, edgecolor='#a5b4fc', alpha=0.6)
    ax.set_title('Cumulative GPA (CGPA) Distribution & Kernel Density Estimate', fontsize=12, fontweight='bold', pad=12)
    ax.set_xlabel('CGPA (0 - 10 Scale)', fontsize=10)
    ax.set_ylabel('Student Count', fontsize=10)
    ax.axvline(df['CGPA'].mean(), color='#ec4899', linestyle='--', linewidth=2, label=f'Mean CGPA: {df["CGPA"].mean():.2f}')
    ax.legend(facecolor='#151c2e', edgecolor='#2a3650')
    plt.tight_layout()
    p1 = os.path.join(PLOTS_DIR, 'cgpa_distribution.png')
    fig.savefig(p1, dpi=160)
    plt.close(fig)
    generated['cgpa_distribution'] = 'plots/cgpa_distribution.png'
    
    # Plot 2: Boxplot of CGPA by Academic Department
    fig, ax = plt.subplots(figsize=(9, 4.8))
    sns.boxplot(x='Department', y='CGPA', data=df, palette='Spectral', ax=ax, width=0.5, boxprops=dict(alpha=0.85))
    sns.stripplot(x='Department', y='CGPA', data=df, color='#f8fafc', size=6, jitter=0.2, ax=ax, alpha=0.7)
    ax.set_title('Academic Performance (CGPA) Across Engineering Departments', fontsize=12, fontweight='bold', pad=12)
    ax.set_xlabel('Engineering Department', fontsize=10)
    ax.set_ylabel('CGPA', fontsize=10)
    plt.xticks(rotation=15)
    plt.tight_layout()
    p2 = os.path.join(PLOTS_DIR, 'department_boxplot.png')
    fig.savefig(p2, dpi=160)
    plt.close(fig)
    generated['department_boxplot'] = 'plots/department_boxplot.png'
    
    # Plot 3: Study Hours vs CGPA Scatter & Regression Trendline
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.regplot(
        x='StudyHoursPerWeek', y='CGPA', data=df,
        scatter_kws={'color': '#06b6d4', 's': 70, 'alpha': 0.8},
        line_kws={'color': '#f59e0b', 'linewidth': 2.5, 'label': 'Trend Line (OLS)'},
        ax=ax
    )
    ax.set_title('Weekly Study Hours vs Cumulative GPA (Correlation Analysis)', fontsize=12, fontweight='bold', pad=12)
    ax.set_xlabel('Independent Study Hours per Week (hrs)', fontsize=10)
    ax.set_ylabel('CGPA', fontsize=10)
    ax.legend(facecolor='#151c2e', edgecolor='#2a3650')
    plt.tight_layout()
    p3 = os.path.join(PLOTS_DIR, 'study_hours_vs_cgpa.png')
    fig.savefig(p3, dpi=160)
    plt.close(fig)
    generated['study_hours_vs_cgpa'] = 'plots/study_hours_vs_cgpa.png'
    
    # Plot 4: Correlation Heatmap of Quantitative Metrics
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    numeric_cols = ['Age', 'Year', 'CGPA', 'AttendanceRate', 'StudyHoursPerWeek', 'MathScore', 'ReadingScore', 'WritingScore']
    corr_matrix = df[numeric_cols].corr()
    sns.heatmap(
        corr_matrix, annot=True, fmt='.2f', cmap='magma',
        cbar=True, ax=ax, square=True, linewidths=0.5, linecolor='#0e1320'
    )
    ax.set_title('Correlation Matrix: Academic & Demographics Features', fontsize=12, fontweight='bold', pad=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    p4 = os.path.join(PLOTS_DIR, 'correlation_heatmap.png')
    fig.savefig(p4, dpi=160)
    plt.close(fig)
    generated['correlation_heatmap'] = 'plots/correlation_heatmap.png'
    
    return generated

if __name__ == '__main__':
    print("[WEEK 3-4] Executing Python, NumPy, Pandas, and Visualization Pipeline...")
    res_py = run_python_essentials_demo()
    print("1. Python OOP Demo: Passed. Total distinction students:", res_py['distinction_count'])
    
    res_np = run_numpy_demo()
    print("2. NumPy Vectorized Ops & Broadcasting: Passed. Matrix shape:", res_np['scores_matrix_shape'])
    
    res_pd = run_pandas_demo()
    print("3. Pandas Merge & GroupBy: Passed. Merged rows:", res_pd['merged_rows_count'])
    
    plots = generate_all_plots()
    print("4. Matplotlib & Seaborn Visualizations Generated successfully:")
    for k, v in plots.items():
        print(f"   - {k}: {v}")
