import openai 
from tqdm import tqdm
from gpt4v_video import GLOBAL_TEMPERATURE

with open('keys/openai.key', 'r') as f:
    openai.api_key = f.readline().strip()


def get_score_openeqa_original(question, answer, pred, model="gpt-4o"):
    response = openai.chat.completions.create(
        model=model,#-turbo",
        temperature=GLOBAL_TEMPERATURE,
        messages=[
                        {
                            "role": "system",
                            "content": 
                            """
                            """
                        },
                        {
                            "role": "user",
                            "content":
                                f"""
You are an AI assistant who will help me to evaluate the response given the question and the correct answer.
To mark a response, you should output a single integer between 1 and 5 (including 1, 5).
5 means that the response perfectly matches the answer.
1 means that the response is completely different from the answer.

Example 1:
Question: Is it overcast?
Answer: no
Response: yes
Your mark: 1

Example 2:
Question: Who is standing at the table?
Answer: woman
Response: Jessica
Your mark: 3

Example 3:
Question: Are there drapes to the right of the bed?
Answer: yes
Response: yes
Your mark: 5

Your Turn:
Question: {question}
Answer: {answer}
Response: {pred}
Your mark:"""
                        }
                    ]
                )
    return response.choices[0].message.content

def get_score_openeqa(question, answer, pred, model="gpt-4o"):
    response = openai.chat.completions.create(
        model=model,#-turbo",
        temperature=GLOBAL_TEMPERATURE,
        messages=[
                        {
                            "role": "system",
                            "content": 
                            """
                            """
                        },
                        {
                            "role": "user",
                            "content":
                                f"""
You are an AI assistant who will help me to evaluate the response given the question and the correct answer. 
To mark a response, you should output only a single integer between 1 and 5 (including 1, 5).
5 means that the response perfectly matches the answer.
1 means that the response is completely different from the answer.

Example 1:
Question: What weather is shown in the video by the beach? 
Answer: It is a beautiful sunny day
Response: It is a typhoon
Your mark: 1

Example 2:
Question: What type of animal is shown on the tree branch?
Answer: It is a squirrel
Response: Rodent
Your mark: 3

Example 3:
Question: What is the young adult doing in the office?
Answer: They are talking to a colleague at their desk
Response: They are working at their desk
Your mark: 4

Example 4:
Question: What is the child doing with their parents in the park?
Answer: The are riding a bike and the parents are cheering them on
Response: Learning to ride a bike
Your mark: 5

Your Turn:
Question: {question}
Answer: {answer}
Response: {pred}
Your mark:"""
                        }
                    ]
                )
    return response.choices[0].message.content


def get_score_lwm(question, answer, pred, model="gpt-4o"):
    response = openai.chat.completions.create(
        model=model,#-turbo",
        temperature=GLOBAL_TEMPERATURE,
        messages=[
                        {
                            "role": "system",
                            "content": 
                                "You are an intelligent chatbot designed for evaluating the correctness of generative outputs for question-answer pairs. "
                                "Your task is to compare the predicted answer with the correct answer and determine if they match meaningfully. Here's how you can accomplish the task:"
                                "------"
                                "##INSTRUCTIONS: "
                                "- Focus on the meaningful match between the predicted answer and the correct answer.\n"
                                "- Consider synonyms or paraphrases as valid matches.\n"
                                "- Evaluate the correctness of the prediction compared to the answer."
                        },
                        {
                            "role": "user",
                            "content":
                                "Please evaluate the following video-based question-answer pair:\n\n"
                                f"Question: {question}\n"
                                f"Correct Answer: {answer}\n"
                                f"Predicted Answer: {pred}\n\n"
                                "Provide your evaluation only as a yes/no and score where the score is an integer value between 0 and 5, with 5 indicating the highest meaningful match. "
                                "Please generate the response in the form of a Python dictionary string with keys 'pred' and 'score', where value of 'pred' is  a string of 'yes' or 'no' and value of 'score' is in INTEGER, not STRING."
                                "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. "
                                "For example, your response should look like this: {'pred': 'yes', 'score': 4.8}."
                        }
                    ]
                )
    return response.choices[0].message.content

# Test example
#question = "What is the capital of France?"
#answer = "Paris"
#pred = "Paris"
#print(get_score(question, answer, pred))

# Get questions from video_files dir
import os

def idx2question(idx):
    try:
        with open(f'{VIDEO_FILES_DIR}/{idx}/questions.txt', 'r') as f:
            question = f.read()
    except:
        with open(f'{VIDEO_FILES_DIR}/{idx}/question.txt', 'r') as f:
            question = f.read()
    return question

def idx2answer(idx):
    try:
        with open(f'{VIDEO_FILES_DIR}/{idx}/answers.txt', 'r') as f:
            answer = f.read()
    except:
        with open(f'{VIDEO_FILES_DIR}/{idx}/answer.txt', 'r') as f:
            answer = f.read()
    return answer

def idx2pred(idx, kind='human'):
    with open(f'{VIDEO_FILES_DIR}/{idx}/{kind}_answer.txt', 'r') as f:
        pred = f.read()
    return pred


if __name__ == "__main__":
    global VIDEO_FILES_DIR
    VIDEO_FILES_DIR = "video_files_20"
    model_output = "32_frames_gpt-4o" # Alternatively, all_frames_gemini-1.5-flash, blind_gpt4o, 16_frames_phi3, 16_frames_gpt-4o, 16_frames_gpt-4-turbo, human
    scoring_model = "gpt-4o" # Alternatively, gpt-4o, gpt-4-turbo, gpt-3.5-turbo
    scores_lwm = []
    accs_lwm = []
    scores_eqa = []
    scores_eqa_orig = []
    answers = []
    preds = []
    questions = []
    num_skipped = 0
    tested_idxs = range(100)
    if False: # Save all model outputs to answers.txt
        pred_str = ""
        for idx in tqdm(tested_idxs):
            pred = idx2pred(idx, kind=model_output)
            print(pred)
            pred_str += str(idx + 1) + ". " + pred

        with open(f'answers.txt', 'w') as f:
            f.write(pred_str)
        exit()

    for idx in tqdm(tested_idxs):
        try:
            question = idx2question(idx)
            answer = idx2answer(idx)
            pred = idx2pred(idx, kind=model_output)
            questions.append(question)
            answers.append(answer)
            preds.append(pred)
            score_lwm = eval(get_score_lwm(question, answer, pred, model=scoring_model))
            #score_eqa = get_score_openeqa(question, answer, pred, model=scoring_model)
            #score_eqa_orig = get_score_openeqa_original(question, answer, pred, model=scoring_model)
            #score_eqa = eval(score_eqa)
            #score_eqa_orig = eval(score_eqa_orig)
        except:
            #idx -= 1
            print("Error with evaluating video", idx, "skipping...")
            num_skipped += 1
            continue # No longer retry the same idx

        if score_lwm['pred'] == 'yes':
            accs_lwm.append(1)
        else:
            accs_lwm.append(0)
        scores_lwm.append(score_lwm['score'])
        #scores_eqa.append(score_eqa)
        #scores_eqa_orig.append(score_eqa_orig)

    print("Final scores for the model:", model_output, "as scored by model:", scoring_model)
    print("Mean score LWM:", sum(scores_lwm)/len(scores_lwm))
    print("Accuracy LWM:", sum(accs_lwm)/len(accs_lwm))
    #print("Mean score EQA (original):", sum(scores_eqa_orig)/len(scores_eqa_orig))
    #print("Mean score EQA (new):", sum(scores_eqa)/len(scores_eqa))
    print("Number of videos skipped:", num_skipped, "out of 100")

    # Save scores per video to file
    with open(f'scores.txt', 'w') as f:
        for idx in range(len(scores_lwm)):
            f.write(f"{idx+1}. LWM Score: {scores_lwm[idx]}, Question: {questions[idx]}, Answer: {answers[idx]}, Prediction: {preds[idx]}\n")
