import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR.parent))

from web_demo.service import _now, _read_json, _write_json


STATE_DIR = BASE_DIR / "data"
JOBS_PATH = STATE_DIR / "training_jobs.json"
LOGS_DIR = STATE_DIR / "logs"


def update_job(job_id: str, **changes):
    jobs = _read_json(JOBS_PATH, [])
    target = None
    for job in jobs:
        if job["id"] == job_id:
            job.update(changes)
            target = job
            break
    _write_json(JOBS_PATH, jobs)
    return target


def main(job_id: str) -> int:
    job = update_job(job_id, status="running", started_at=_now(), message="训练任务执行中。")
    if not job:
        return 1
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGS_DIR / f"{job_id}.log"
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(f"[{_now()}] start training job {job_id}\n")
        log_file.write("command: " + " ".join(job["command"]) + "\n\n")
        log_file.flush()
        result = subprocess.run(job["command"], cwd=str(BASE_DIR.parent), stdout=log_file, stderr=subprocess.STDOUT)
    if result.returncode == 0:
        update_job(job_id, status="completed", finished_at=_now(), message="训练完成。")
    else:
        update_job(job_id, status="failed", finished_at=_now(), message=f"训练失败，退出码 {result.returncode}。")
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
