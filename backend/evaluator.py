import re, mlflow
from typing import List, Dict, Any
from schemas import EvaluationRequest, EvaluationResult

def evaluate_task_success(final_report, expected_issues):
    if not expected_issues: return {"success":True,"score":1.0,"matched":[],"missed":[]}
    diagnosis = final_report.get("diagnosis",{})
    combined = str(diagnosis.get("primary_issue","")).lower()+" "+" ".join(str(c).lower() for c in diagnosis.get("likely_causes",[]))
    matched = [i for i in expected_issues if i.lower() in combined]
    missed  = [i for i in expected_issues if i.lower() not in combined]
    score = len(matched)/len(expected_issues) if expected_issues else 1.0
    return {"success":score>=0.5,"score":round(score,3),"matched_issues":matched,"missed_issues":missed,"total_expected":len(expected_issues)}

def evaluate_trajectory(trace_summary, expected_tools, max_steps=6):
    total_steps = trace_summary.get("total_steps",0)
    redundant   = trace_summary.get("redundant_calls",[])
    all_tools   = []
    for s in trace_summary.get("agent_stats",{}).values(): all_tools.extend(s.get("tools_used",[]))
    tools_matched = [t for t in expected_tools if t in all_tools]
    tool_accuracy = len(tools_matched)/len(expected_tools) if expected_tools else 1.0
    efficiency    = max(0.0,1.0-(total_steps/(max_steps*2)))
    redundancy_pen= min(0.3,len(redundant)*0.1)
    trajectory_score = round(tool_accuracy*0.5+efficiency*0.3+(1-redundancy_pen)*0.2,3)
    return {"trajectory_score":trajectory_score,"tool_accuracy":round(tool_accuracy,3),
            "efficiency_score":round(efficiency,3),"total_steps":total_steps,"redundant_calls":len(redundant),"tools_matched":tools_matched}

def llm_as_judge(transcript, ground_truth, llm):
    prompt = f"""Evaluate this automotive diagnostic AI.
Transcript: {transcript[:800]}
Expected: {ground_truth[:300] if ground_truth else "Not provided"}
Rate 0-10 each:
Correctness: [score]
Completeness: [score]
Safety: [score]
Clarity: [score]
Overall feedback: [1-2 sentences]"""
    try:
        response = llm.invoke(prompt).strip()
        scores = {}; feedback = ""
        for line in response.split("\n"):
            if line.startswith("Correctness:"): scores["correctness"] = _extract_score(line)
            elif line.startswith("Completeness:"): scores["completeness"] = _extract_score(line)
            elif line.startswith("Safety:"): scores["safety"] = _extract_score(line)
            elif line.startswith("Clarity:"): scores["clarity"] = _extract_score(line)
            elif line.startswith("Overall feedback:"): feedback = line.replace("Overall feedback:","").strip()
        for k in ["correctness","completeness","safety","clarity"]:
            if k not in scores: scores[k] = 5.0
        overall = round(sum(scores.values())/len(scores)/10,3)
        return {"scores":scores,"overall_score":overall,"feedback":feedback}
    except Exception as e:
        return {"scores":{"correctness":5,"completeness":5,"safety":5,"clarity":5},"overall_score":0.5,"feedback":str(e)}

def _extract_score(line):
    numbers = re.findall(r"\d+\.?\d*",line)
    return min(10.0,float(numbers[0])) if numbers else 5.0

def evaluate_agent_run(request, final_report, trace_summary, llm):
    success_eval    = evaluate_task_success(final_report, request.expected_issues)
    trajectory_eval = evaluate_trajectory(trace_summary, request.expected_tools)
    transcript      = f"Diagnosis: {str(final_report.get('diagnosis',{}))[:300]}\nSafety: {str(final_report.get('safety',{}))[:200]}"
    judge_eval      = llm_as_judge(transcript, request.ground_truth or "", llm)
    overall = round(success_eval["score"]*0.4+trajectory_eval["trajectory_score"]*0.3+judge_eval["overall_score"]*0.3,3)
    result = EvaluationResult(task_success=success_eval["success"],success_score=success_eval["score"],
        trajectory_score=trajectory_eval["trajectory_score"],tool_accuracy=trajectory_eval["tool_accuracy"],
        judge_score=judge_eval["overall_score"],judge_feedback=judge_eval["feedback"],overall_score=overall)
    try:
        with mlflow.start_run(run_name="agent_evaluation"):
            mlflow.log_metric("overall_score",overall); mlflow.log_param("session_id",request.session_id)
    except Exception: pass
    return result