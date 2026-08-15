import mlflow
from db import insert_feedback, get_feedback_stats as db_stats, get_feedback_by_session

def log_feedback(query, answer, rating, comment="", session_id=""):
    entry = insert_feedback(query, answer, rating, comment, session_id)
    try:
        with mlflow.start_run(run_name="user_feedback"):
            mlflow.log_param("rating", "positive" if rating else "negative")
            mlflow.log_metric("is_positive", 1 if rating else 0)
    except Exception:
        pass
    return entry

def get_feedback_stats():
    return db_stats()

def get_session_feedback(session_id):
    return get_feedback_by_session(session_id)