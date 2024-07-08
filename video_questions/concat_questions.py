file_list = [
    "video_questions_n=10_subset.txt",
    "video_questions_n=10-start=100.txt",
    "video_questions_n=10-start=116.txt",
    # TODO: Run generate_video_questions again with start_idx=4030, add final file here
]

all_questions = []

for file in file_list:
    with open(file, 'r') as f:
        video_questions = eval(f.read())
        all_questions.extend(video_questions)

with open(f'video_questions_n=10.txt', 'w') as f:
    f.write(str(all_questions) + '\n')