import threading
from datetime import datetime
from typing import Dict, List, Optional, Any
from schemas import BlackboardEntry, TaskType, SeverityLevel

class Blackboard:
    def __init__(self):
        self._store: Dict[str, BlackboardEntry] = {}
        self._lock = threading.Lock()
        self._log: List[Dict] = []

    def write(self, agent_name, task_type, content, confidence=0.5):
        with self._lock:
            key = task_type.value
            if key in self._store:
                existing = self._store[key]
                if existing.confidence >= confidence and existing.is_final:
                    self._log.append({"event":"REDUNDANT_WRITE_BLOCKED","agent":agent_name,"task_type":task_type.value,"timestamp":datetime.now().isoformat()})
                    return False
            entry = BlackboardEntry(agent_name=agent_name,task_type=task_type,content=content,confidence=confidence,timestamp=datetime.now().isoformat(),is_final=True)
            self._store[key] = entry
            self._log.append({"event":"WRITE","agent":agent_name,"task_type":task_type.value,"timestamp":entry.timestamp})
            return True

    def read(self, task_type) -> Optional[BlackboardEntry]:
        with self._lock: return self._store.get(task_type.value)

    def read_all(self) -> Dict[str, BlackboardEntry]:
        with self._lock: return dict(self._store)

    def is_task_done(self, task_type) -> bool:
        with self._lock:
            entry = self._store.get(task_type.value)
            return entry is not None and entry.is_final

    def get_audit_log(self):
        with self._lock: return list(self._log)

    def clear(self):
        with self._lock: self._store.clear(); self._log.clear()

    def summary(self):
        with self._lock:
            return {"entries":list(self._store.keys()),"total_entries":len(self._store),
                    "agents_wrote":list(set(e.agent_name for e in self._store.values())),"log_entries":len(self._log)}