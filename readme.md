# Long Context Video Benchmark

A benchmark for evaluating video understanding models on longer context videos.

## Setup

1. Create and activate the conda environment:

   ```bash
   conda env create -f env.yml
   conda activate lc_benchmark
   ```

2. Download and prepare the dataset:

- Generate questions using `generate_video_questions.py` which creates `video_questions/video_questions_n=10.txt`
- Download source videos by running `python download_videos.py`
- Concatenate videos into longer sequences using `python concat_mp4s.py` 
- Extract video frames by running `python extract_frames.py`
- [NEW] Apply transitions to frames by running `python extract_frames.py --transition_type <transition_type>`

3. Run evaluation:

- Execute baseline model: `python run_baseline.py --model <model_name>`
- Evaluate results: `python eval_answers.py --model <model_name>`

## Dataset Structure

The benchmark uses concatenated video sequences with associated questions. The dataset includes:

- Original source videos
- Concatenated longer video sequences 
- Extracted video frames
- Question-answer pairs in `video_questions_n=10.txt`


## Known Issues

Some examples have been identified as problematic and require updates:

- Questions with unclear references (e.g., "which is budapest?")
- Questions about specific locations that may be ambiguous
- Questions with incorrect assumptions about video content
- Questions with potentially biased language

## Problematic Examples

### Example 29
- Issue: Asks for timelapse but shows 3 videos??

### Example 30
- Issue: Unclear which is Budapest?

### Example 35
- Issue: Toronto?

### Example 45
- **Original Question:** What was the girl doing before she answered her cellphone?
- **Suggested Question:** What was the girl doing before the woman answered her cellphone?

### Example 47
- **Original Question:** What activities did the two men perform on the beach?
- **Suggested Question:** What activities did the two men perform together in the snow?
- **Original Answer:** They built a snowman and played snowball volleyball.
- **Suggested Answer:** They threw the snow to each other and built a snowman.

### Example 48
- Issue: Remove references to Kyiv, Ukraine. General trend.

### Example 55
- Issue: Remove the word "hispanic"

### Example 59
- Issue: Not answerable?

### Example 63
- **Original Question:** What was the gift box in one of the videos representing?
- **Suggested Question:** What was the gift box in one of the videos full of?
- **Original Answer:** The concept of Merry Christmas and Happy New Year.
- **Suggested Answer:** It is full of money, specifically $100 bills.

### Example 65
- Issue: Change "hotel" to "resort"?

### Example 66
- **Original Question:** Did the person see any wildlife during the video sequence?
- **Suggested Question:** Was there any wildlife during the video sequence?

### Example 71
- Issue: Bird species are hard for humans to answer

### Example 80
- **Original Question:** What can you do to save energy during the winter?
- **Suggested Question:** What is shown as a way to save energy?
- **Original Answer:** Adjusting a wall thermostat from 68°F to 62°F can save energy during the winter.
- **Suggested Answer:** Adjusting a wall thermostat to better match the outdoor temperature.

### Example 88
- **Original Question:** What type of landscape was shown in the video from Washington, United States of America?
- **Suggested Question:** What type of landscape was shown in the video containing snow?

### Example 89
- **Original Question:** "young pople"
- **Suggested Question:** "young adults"