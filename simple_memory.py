import sqlite3
import json
from datetime import datetime
from typing import Dict, List, Any

class SimpleMemoryManager:
    def __init__(self):
        self.db = sqlite3.connect('memory.db', check_same_thread=False)
        self.db.execute('''CREATE TABLE IF NOT EXISTS tasks 
                          (id INTEGER PRIMARY KEY, agent_id TEXT, task TEXT, result TEXT, created_at TEXT)''')
        self.db.commit()
    
    def store_task_result(self, agent_id: str, task: str, result: Dict[str, Any], **kwargs) -> str:
        cursor = self.db.execute('INSERT INTO tasks (agent_id, task, result, created_at) VALUES (?, ?, ?, ?)',
                                (agent_id, task, json.dumps(result), datetime.now().isoformat()))
        self.db.commit()
        return str(cursor.lastrowid)
    
    def get_task_history(self, agent_id: str = None, limit: int = 50) -> List[Dict[str, Any]]:
        if agent_id:
            cursor = self.db.execute('SELECT * FROM tasks WHERE agent_id = ? ORDER BY created_at DESC LIMIT ?', (agent_id, limit))
        else:
            cursor = self.db.execute('SELECT * FROM tasks ORDER BY created_at DESC LIMIT ?', (limit,))
        
        return [{'id': row[0], 'agent_id': row[1], 'task_description': row[2], 'result': json.loads(row[3]), 'created_at': row[4]} 
                for row in cursor.fetchall()]
    
    def get_memory_statistics(self, agent_id: str = None) -> Dict[str, Any]:
        cursor = self.db.execute('SELECT COUNT(*) FROM tasks')
        total = cursor.fetchone()[0]
        return {'task_statistics': {'total_tasks': total}, 'memory_by_type': {}, 'learning_statistics': {}}