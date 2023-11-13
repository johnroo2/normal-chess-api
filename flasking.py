from flask import Flask, jsonify, request
from predict import computer_instance
from flask_cors import CORS, cross_origin
app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route("/", methods=['POST'])
@cross_origin()
def index():
    payload = dict(request.json)
    prediction = computer_instance.predict(
        payload["board"], payload["col"],
        payload["castle"], payload["passant"],
        payload["halfmoves"], payload["fullmoves"]
    )
    return jsonify({"prediction": prediction, "to_move": payload["col"], "board": payload["board"]})

if __name__ == '__main__':
    app.run(port=8000, debug=True)