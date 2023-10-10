import json
import sqlite3
from datetime import datetime


class TaskDatabase:
    def __init__(self, db_path):
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self.create_table_if_not_exists()

    def create_table_if_not_exists(self):
        self.cursor.execute('''CREATE TABLE IF NOT EXISTS tasks (
                                id INTEGER PRIMARY KEY AUTOINCREMENT,
                                task TEXT,
                                task_config JSON,
                                workspace TEXT,
                                dataset TEXT,
                                deployment TEXT,
                                created_at DATETIME,
                                last_updated DATETIME
                            )''')
        self.conn.commit()

    def close(self):
        self.conn.close()

    def create_task(self, task_data):
        task, task_config, workspace, dataset, deployment = task_data['task'], task_data['task_config'], task_data['workspace'], task_data['dataset'], task_data['deployment']
        created_at = datetime.now()
        last_updated = created_at
        self.cursor.execute('''INSERT INTO tasks (task, task_config, workspace, dataset, deployment, created_at, last_updated)
                               VALUES (?, ?, ?, ?, ?, ?, ?)''',
                            (task, json.dumps(task_config), workspace, dataset, deployment, created_at, last_updated))
        self.conn.commit()
        return self.cursor.lastrowid

    def read_task(self, task_id):
        self.cursor.execute('''SELECT * FROM tasks WHERE id = ?''', (task_id,))
        task_data = self.cursor.fetchone()
        if task_data:
            task_id, task, task_config_json, workspace, dataset, deployment, created_at, last_updated = task_data
            task_config = json.loads(task_config_json)
            return {
                'id': task_id,
                'task': task,
                'task_config': task_config,
                'workspace': workspace,
                'dataset': dataset,
                'deployment': deployment,
                'created_at': created_at,
                'last_updated': last_updated
            }
        else:
            return None

    def update_task(self, task_id, new_task_data):
        task, task_config, workspace, dataset, deployment = new_task_data['task'], new_task_data['task_config'], new_task_data['workspace'], new_task_data['dataset'], new_task_data['deployment']
        last_updated = datetime.now()
        self.cursor.execute('''UPDATE tasks
                               SET task = ?, task_config = ?, workspace = ?, dataset = ?, deployment = ?, last_updated = ?
                               WHERE id = ?''',
                            (task, json.dumps(task_config), workspace, dataset, deployment, last_updated, task_id))
        self.conn.commit()

    def delete_task(self, task_id):
        self.cursor.execute('''DELETE FROM tasks WHERE id = ?''', (task_id,))
        self.conn.commit()

    def get_task_ids_by_query(self, query_dict):
        query = "SELECT id FROM tasks WHERE "
        query_values = []

        for key, value in query_dict.items():
            if key != "task_config":
                query += f"{key} = ? AND "
                query_values.append(value)

        # Remove the trailing 'AND ' from the query
        query = query[:-5]

        self.cursor.execute(query, tuple(query_values))
        task_ids = [row[0] for row in self.cursor.fetchall()]

        return task_ids