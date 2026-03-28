"""
Resume Screening AI — Flask REST API
=====================================
Wraps the core screener as HTTP endpoints.

Install : pip install flask flask-cors spacy scikit-learn pandas numpy
          python -m spacy download en_core_web_sm
Run     : python api.py   ->  http://127.0.0.1:5000

Endpoints
---------
GET  /             welcome + docs
GET  /health       {"status":"ok"}
GET  /demo         run built-in sample data
POST /screen       screen your own resumes
"""

import logging
from resume_screener import ResumeScreener, JD, RESUMES

from flask      import Flask, request, jsonify
from flask_cors import CORS

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)


def _serialise(screener):
    return [{
        "rank":             r.rank,
        "name":             r.name,
        "score":            r.score,
        "score_pct":        round(r.score * 100, 1),
        "shortlisted":      r.shortlisted,
        "experience_years": r.experience_years,
        "skills":           sorted(r.skills),
        "email":            r.email,
        "phone":            r.phone,
    } for r in screener.resumes]


@app.get("/")
def index():
    return jsonify({
        "service": "Resume Screening AI",
        "status":  "running",
        "endpoints": {
            "GET  /":        "this page",
            "GET  /health":  "health check",
            "GET  /demo":    "run sample data",
            "POST /screen":  "screen your resumes",
        },
    })


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.get("/demo")
def demo():
    s = ResumeScreener(threshold=0.15)
    s.load_jd(JD)
    for r in RESUMES:
        s.add_resume(r["name"], r["text"])
    s.screen()
    return jsonify({
        "total":       len(RESUMES),
        "shortlisted": sum(1 for r in s.resumes if r.shortlisted),
        "results":     _serialise(s),
    })


@app.post("/screen")
def screen():
    body = request.get_json(silent=True)
    if not body:
        return jsonify({"error": "Body must be valid JSON",
                        "hint":  "Set header Content-Type: application/json"}), 400

    jd   = body.get("job_description", "").strip()
    cvs  = body.get("resumes", [])
    if not jd:  return jsonify({"error": "job_description required"}), 400
    if not cvs: return jsonify({"error": "resumes list required"}),    400

    s = ResumeScreener(
        threshold=float(body.get("threshold", 0.15)),
        top_n    =int  (body.get("top_n",     0)),
    )
    s.load_jd(jd)
    loaded = 0
    for item in cvs:
        txt = str(item.get("text", "")).strip()
        if txt:
            s.add_resume(str(item.get("name", "Unknown")), txt)
            loaded += 1

    if not loaded:
        return jsonify({"error": "No usable resume text found"}), 400

    s.screen()
    return jsonify({
        "total":       loaded,
        "shortlisted": sum(1 for r in s.resumes if r.shortlisted),
        "results":     _serialise(s),
    })


@app.errorhandler(404)
def e404(_): return jsonify({"error": "Not found", "hint": "GET / for docs"}), 404

@app.errorhandler(405)
def e405(_): return jsonify({"error": f"Method not allowed on {request.path}"}), 405

@app.errorhandler(500)
def e500(e): log.exception("500"); return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print()
    print("  Resume Screening AI  -  Flask API")
    print("  Home   -> http://127.0.0.1:5000/")
    print("  Demo   -> http://127.0.0.1:5000/demo")
    print("  Health -> http://127.0.0.1:5000/health")
    print()
    app.run(debug=True, host="127.0.0.1", port=5000)
