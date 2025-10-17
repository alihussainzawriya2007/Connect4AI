from flask import Flask, render_template, request, jsonify


app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/played_move", methods=["POST"])
def played_move():
    # Accept JSON or form-encoded data
    data = request.get_json(silent=True)
    if data is None:
        data = request.form.to_dict()
    # TODO: handle the played move (update game state, etc.)
    return jsonify({"status": "ok", "received": data})


@app.route("/get_move", methods=["GET"])
def get_move():
    # TODO: compute or fetch an AI move; here we return a placeholder move
    to_move = None
    return jsonify(to_move)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)