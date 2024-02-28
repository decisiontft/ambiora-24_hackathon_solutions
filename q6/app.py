from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Dummy database to store user profiles
users = {}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/create_profile', methods=['POST'])
def create_profile():
    data = request.json
    username = data['username']
    password = data['password']
    users[username] = password
    return jsonify({'message': 'Profile created successfully'})

@app.route('/upload_file', methods=['POST'])
def upload_file():
    # Handle file upload logic here
    return jsonify({'message': 'File uploaded successfully'})

if __name__ == '__main__':
    app.run(debug=True)
