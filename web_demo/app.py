from pathlib import Path

from flask import Flask, render_template, request, send_from_directory

from .service import OpenVocabularyDefectSystem


BASE_DIR = Path(__file__).resolve().parent
app = Flask(__name__, template_folder=str(BASE_DIR / "templates"), static_folder=str(BASE_DIR / "static"))
system = OpenVocabularyDefectSystem(BASE_DIR)


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None
    defect_terms = "划痕, 裂纹, 污渍, 孔洞"
    scene_name = "冲压件产线-A"

    if request.method == "POST":
        image = request.files.get("image")
        defect_terms = request.form.get("defect_terms", defect_terms)
        scene_name = request.form.get("scene_name", scene_name)
        if not image or not image.filename:
            error = "请先上传待检测图像。"
        else:
            terms = [item.strip() for item in defect_terms.replace("，", ",").split(",") if item.strip()]
            result = system.analyze(image, terms, scene_name)

    return render_template(
        "index.html",
        result=result,
        error=error,
        defect_terms=defect_terms,
        scene_name=scene_name,
        history=system.load_history(),
        runtime_mode=system.runtime_mode,
    )


@app.route("/uploads/<path:filename>")
def uploads(filename: str):
    return send_from_directory(system.upload_dir, filename)


@app.route("/generated/<path:filename>")
def generated(filename: str):
    return send_from_directory(system.output_dir, filename)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=True)
