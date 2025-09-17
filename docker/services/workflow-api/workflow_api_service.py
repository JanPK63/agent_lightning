from flask import Flask, jsonify
import os

app = Flask(__name__)

@app.route('/health')
def health():
    return jsonify({"status": "healthy", "service": "workflow-api"})

@app.route('/workflows')
def workflows():
    return jsonify({"workflows": []})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8004)