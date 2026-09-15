import unittest
import json
from app import app, load_students, save_students

class TestSIMSAPI(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_01_get_students(self):
        response = self.app.get('/api/students')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertTrue(data['success'])
        self.assertGreater(data['count'], 0)

    def test_02_filter_students(self):
        response = self.app.get('/api/students?branch=Computer%20Science&year=3')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        for s in data['students']:
            self.assertEqual(s['branch'], 'Computer Science')
            self.assertEqual(s['year'], 3)

    def test_03_get_single_student(self):
        response = self.app.get('/api/students/STU001')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertTrue(data['success'])
        self.assertEqual(data['student']['id'], 'STU001')

    def test_04_add_update_delete_student(self):
        # 1. Add student
        new_payload = {
            "name": "Automated Test Student",
            "email": "autotest@college.edu",
            "branch": "Data Science",
            "year": 2,
            "cgpa": 8.95,
            "attendance": 90.0,
            "city": "Pune",
            "skills": ["Python", "Flask", "Unit Testing"]
        }
        create_res = self.app.post(
            '/api/students',
            data=json.dumps(new_payload),
            content_type='application/json'
        )
        self.assertEqual(create_res.status_code, 201)
        created_data = json.loads(create_res.data)
        new_id = created_data['student']['id']
        self.assertTrue(new_id.startswith('STU'))

        # 2. Update student
        update_payload = {"cgpa": 9.25, "status": "Active"}
        update_res = self.app.put(
            f'/api/students/{new_id}',
            data=json.dumps(update_payload),
            content_type='application/json'
        )
        self.assertEqual(update_res.status_code, 200)
        updated_data = json.loads(update_res.data)
        self.assertEqual(updated_data['student']['cgpa'], 9.25)

        # 3. Delete student
        del_res = self.app.delete(f'/api/students/{new_id}')
        self.assertEqual(del_res.status_code, 200)

        # 4. Verify 404 after deletion
        check_res = self.app.get(f'/api/students/{new_id}')
        self.assertEqual(check_res.status_code, 404)

    def test_05_stats(self):
        response = self.app.get('/api/stats')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertTrue(data['success'])
        self.assertIn('total_students', data['stats'])
        self.assertIn('average_cgpa', data['stats'])
        self.assertIn('branch_distribution', data['stats'])

    def test_06_dataset_info(self):
        response = self.app.get('/api/dataset-info')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertTrue(data['success'])
        self.assertIn('Kaggle', data['dataset']['name'])

if __name__ == '__main__':
    unittest.main()
