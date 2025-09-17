from flask import Flask, jsonify
import os

app = Flask(__name__)

@app.route('/health')
def health():
    return jsonify({"status": "healthy", "service": "rl-api"})

@app.route('/training')
def training():
    return jsonify({"training_sessions": []})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8003)