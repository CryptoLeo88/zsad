from pathlib import Path

from flask import Flask, redirect, render_template, request, send_from_directory, url_for

from .service import OpenVocabularyDefectSystem


BASE_DIR = Path(__file__).resolve().parent
app = Flask(__name__, template_folder=str(BASE_DIR / "templates"), static_folder=str(BASE_DIR / "static"))
system = OpenVocabularyDefectSystem(BASE_DIR)


@app.context_processor
def inject_global_data():
    dashboard = system.dashboard_data()
    return {
        "runtime_mode": system.runtime_mode,
        "sidebar_counts": {
            "jobs": dashboard["job_count"],
            "history": dashboard["history_count"],
            "vocab": dashboard["vocab_count"],
        },
    }


@app.route("/")
def dashboard():
    return render_template("dashboard.html", active_page="dashboard", dashboard=system.dashboard_data())


@app.route("/inspect", methods=["GET", "POST"])
def inspect():
    result = None
    error = None
    form_data = {
        "scene_name": "冲压件产线-A",
        "defect_terms": "，".join(system.vocab_store.term_names()),
        "selected_model_path": "",
        "selected_model_option": "",
        "selected_framework": "开放词汇通用检测框架",
        "selected_backbone": "ViT-L/14@336px",
    }
    if request.method == "POST":
        image = request.files.get("image")
        form_data["scene_name"] = request.form.get("scene_name", form_data["scene_name"])
        form_data["defect_terms"] = request.form.get("defect_terms", form_data["defect_terms"])
        form_data["selected_model_option"] = request.form.get("selected_model_option", "")
        manual_model_path = request.form.get("selected_model_path", "")
        form_data["selected_model_path"] = manual_model_path or form_data["selected_model_option"]
        form_data["selected_framework"] = request.form.get("selected_framework", form_data["selected_framework"])
        form_data["selected_backbone"] = request.form.get("selected_backbone", form_data["selected_backbone"])
        if not image or not image.filename:
            error = "请先上传待检测图像。"
        else:
            terms = [item.strip() for item in form_data["defect_terms"].replace("，", ",").split(",") if item.strip()]
            result = system.analyze(
                image,
                terms,
                form_data["scene_name"],
                form_data["selected_model_path"],
                form_data["selected_framework"],
                form_data["selected_backbone"],
            )
    return render_template(
        "inspect.html",
        active_page="inspect",
        result=result,
        error=error,
        form_data=form_data,
        available_models=system.available_models(),
    )


@app.route("/training", methods=["GET", "POST"])
def training():
    defaults = system.training_defaults()
    if request.method == "POST":
        form_data = {key: request.form.get(key, value) for key, value in defaults.items()}
        dataset_file = request.files.get("dataset_archive")
        if dataset_file and dataset_file.filename:
            form_data["train_data_path"] = system.save_uploaded_dataset(dataset_file, form_data["job_name"] or "dataset")
        job_id = system.training.start_job(form_data)
        return redirect(url_for("training_detail", job_id=job_id))
    return render_template(
        "training.html",
        active_page="training",
        profiles=system.training.profiles(),
        jobs=system.training.list_jobs(),
        form_data=defaults,
    )


@app.route("/training/<job_id>")
def training_detail(job_id: str):
    job = system.training.get_job(job_id)
    if not job:
        return redirect(url_for("training"))
    return render_template(
        "training_detail.html",
        active_page="training",
        job=job,
        log_text=system.training.tail_log(job_id),
        jobs=system.training.list_jobs()[:8],
    )


@app.route("/vocabulary", methods=["GET", "POST"])
def vocabulary():
    if request.method == "POST":
        action = request.form.get("action", "add")
        if action == "add":
            term = request.form.get("term", "").strip()
            if term:
                system.vocab_store.add_term(
                    term,
                    request.form.get("category", ""),
                    request.form.get("description", ""),
                )
        elif action == "delete":
            system.vocab_store.remove_term(request.form.get("term_id", ""))
        return redirect(url_for("vocabulary"))
    return render_template("vocabulary.html", active_page="vocabulary", terms=system.vocab_store.list_terms())


@app.route("/history")
def history():
    return render_template(
        "history.html",
        active_page="history",
        items=system.load_history(),
        stats=system.history_stats(),
    )


@app.route("/uploads/<path:filename>")
def uploads(filename: str):
    return send_from_directory(system.upload_dir, filename)


@app.route("/generated/<path:filename>")
def generated(filename: str):
    return send_from_directory(system.output_dir, filename)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=True)
