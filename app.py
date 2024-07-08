from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os

app = Flask(__name__)
BASE_DIR = "video_files"

@app.route('/')
def index():
    dirs = [d for d in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, d))]
    return render_template('index.html', directories=dirs)

@app.route('/directory/<int:dir_id>')
def directory(dir_id):
    dir_path = os.path.join(BASE_DIR, str(dir_id))
    videos = [f for f in os.listdir(dir_path) if f.endswith('.mp4')]
    questions = ""
    # try questions.txt first, then question.txt
    try:
        with open(os.path.join(dir_path, 'questions.txt'), 'r') as file:
            questions = file.read()
    except:
        with open(os.path.join(dir_path, 'question.txt'), 'r') as file:
            questions = file.read()
    return render_template('directory.html', dir_id=dir_id, videos=videos, questions=questions)

@app.route('/video/<int:dir_id>/<filename>')
def video(dir_id, filename):
    dir_path = os.path.join(BASE_DIR, str(dir_id))
    return send_from_directory(dir_path, filename)

@app.route('/save_answer/<int:dir_id>', methods=['POST'])
def save_answer(dir_id):
    answer = request.form['answer']
    dir_path = os.path.join(BASE_DIR, str(dir_id))
    with open(os.path.join(dir_path, 'human_answer.txt'), 'w') as file:
        file.write(answer)
    return redirect(url_for('directory', dir_id=dir_id+1)) # move to next dir

if __name__ == '__main__':
    app.run(debug=True)
