file_list = [
    "video_questions_action_n=10-start=0.txt",
    "video_questions_action_n=10-start=1.txt",
]

output_file = "video_questions_action_n=10.txt"

all_questions = []

for file in file_list:
    with open(file, 'r') as f:
        video_questions = eval(f.read())
        all_questions.extend(video_questions)

with open(output_file, 'w') as f:
    f.write(str(all_questions) + '\n')