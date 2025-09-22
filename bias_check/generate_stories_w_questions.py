import os
import random
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import anthropic
import hydra
import pandas as pd
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from openai import OpenAI


def get_folder_path_name(cfg: OmegaConf):
    try:
        run_idx = HydraConfig.get().job.num
    except Exception:
        run_idx = int(os.environ.get('HYDRA_JOB_NUM', 0))

    return cfg.timestamp + '_' + str(run_idx)


def model_client(model_name):
    if 'gpt-5' == model_name:
        client = OpenAI(base_url='https://api.openai.com/v1')
    elif 'claude-sonnet-4-20250514' == model_name:
        client = anthropic.Anthropic()
    else:
        client = OpenAI(
            base_url='https://openrouter.ai/api/v1',
            api_key=os.environ.get('OPENROUTER_API_KEY'),
        )

    return client


def llm_call(model, prompt_text):
    if 'gpt-5' == model:
        client = model_client(model)
        response = client.responses.create(
            model=model,
            input=[
                {
                    'role': 'developer',
                    'content': "You are a helpful AI assistant that doesn't make mistakes.",
                },
                {'role': 'user', 'content': prompt_text},
            ],
        )
        reply = response.output_text

    else:
        client = model_client(model)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    'role': 'developer',
                    'content': "You are a helpful AI assistant that doesn't make mistakes.",
                },
                {'role': 'user', 'content': prompt_text},
            ],
        )
        reply = response.choices[0].message.content
    return reply


def _clean_text(s: str) -> str:
    return re.sub(r'\s+', ' ', s or '').strip()


def _normalize_correct_answer(s: str) -> str:
    """
    Extract a single letter A-D from values like:
    'A', 'A)', '[A)', 'Correct: A)', etc.
    Falls back to cleaned raw text if no A-D found.
    """
    s = _clean_text(s)
    m = re.search(r'\b([A-Da-d])\s*[\)\].]?\b', s)
    return m.group(1).upper() if m else s


def parse_story_and_questions(raw: str) -> dict:
    """
    Returns:
      {
        "story": "...",
        "questions": [
          {
            "id": "1",
            "text": "...",
            "answers": ["A) ...", "B) ...", "C) ...", "D) ..."],
            "correct_answer": "B",
            "source": "..."
          },
          ...
        ]
      }
    """
    # Make it well-formed by wrapping in a root
    wrapped = f'<root>{raw}</root>'
    root = ET.fromstring(wrapped)

    # Story
    story_el = root.find('story')
    story = _clean_text(story_el.text if story_el is not None else '')

    # Questions
    questions_el = root.find('questions')
    questions: list[dict] = []
    if questions_el is not None:
        for q in questions_el.findall('question'):
            qid = q.attrib.get('id', '')

            # Question text
            qtext_el = q.find('text')
            qtext = _clean_text(qtext_el.text if qtext_el is not None else '')

            # Answers in order answer1..answer4
            answers = []
            for i in range(1, 5):
                a_el = q.find(f'answer{i}')
                if a_el is not None and (a_el.text or '').strip():
                    answers.append(_clean_text(a_el.text))

            # Correct answer (normalize to A-D)
            ca_el = q.find('correct_answer')
            correct_answer = _normalize_correct_answer(
                ca_el.text if ca_el is not None else ''
            )

            # Source
            src_el = q.find('source')
            source = _clean_text(src_el.text if src_el is not None else '')

            questions.append(
                {
                    'id': qid,
                    'text': qtext,
                    'answers': answers,
                    'correct_answer': correct_answer,
                    'source': source,
                }
            )

    return {'story': story, 'questions': questions}


def build_prompt(question_answers, story):
    blocks = []

    # Instructions: compact answers only
    instructions = (
        'You are given a story and multiple-choice questions about it.\n'
        'Return ONLY this block with one line per question as <id>=<A/B/C/D>:\n'
        '<answers>\n'
        '1=A\n'
        '2=B\n'
        '</answers>\n'
        'Do NOT change anything else.'
        "If you don't know the answer, nevertheless pick one of A, B, C, or D."
    )

    # Story section at the top
    blocks.append('=== STORY ===')
    blocks.append(story.strip())
    blocks.append('=== END STORY ===\n')

    # Questions (keep as-is for context)
    for item in question_answers:
        q_lines = []
        q_lines.append('=== QUESTION ===')
        q_lines.append(f'ID: {item["id"]}')
        q_lines.append(f'TEXT: {item["text"]}')

        labels = ['A', 'B', 'C', 'D']
        for lab, opt in zip(labels, item['answers'][:4]):
            cleaned = opt.strip()
            if cleaned.startswith(f'{lab}) '):
                cleaned = cleaned[len(f'{lab}) ') :]
            q_lines.append(f'OPTION {lab}: {cleaned}')

        # No placeholder line (saves tokens in the model's output)
        q_lines.append('=== END ===')
        blocks.append('\n'.join(q_lines))

    return instructions + '\n\n' + '\n\n'.join(blocks)


def parse_llm_answers(output_text: str, question_answers: list[dict]) -> dict[str, str]:
    """
    Returns {question_id: 'A'|'B'|'C'|'D' or None}.
    Priority:
      1) Compact <answers> block with lines: "<id>=<A/B/C/D>"
      2) Full blocks with ID + 'YOUR ANSWER: X'
      3) Bare 'YOUR ANSWER: X' lines mapped by order
    """
    text = output_text.replace('\r\n', '\n').replace('\r', '\n')
    ids_in_order = [q['id'] for q in question_answers]

    # --- 1) Compact <answers> block ---
    # Try to isolate the block; if not present, consider the whole text.
    m = re.search(
        r'<answers>\s*(.*?)\s*</answers>', text, flags=re.DOTALL | re.IGNORECASE
    )
    block = m.group(1) if m else text

    # Pattern: "id = Letter"
    pairs = re.findall(
        r'^\s*([0-9A-Za-z_-]+)\s*=\s*([ABCD])\s*$',
        block,
        flags=re.MULTILINE | re.IGNORECASE,
    )
    ans_map = {qid: ans.upper() for qid, ans in pairs}

    # Also accept verbose lines like "Question ID: 2: Answer: A"
    if not ans_map:
        pairs2 = re.findall(
            r'^\s*(?:Question\s*ID:\s*)?([0-9A-Za-z_-]+)\s*[:=]\s*(?:Answer:\s*)?([ABCD])\s*$',
            block,
            flags=re.MULTILINE | re.IGNORECASE,
        )
        ans_map = {qid: ans.upper() for qid, ans in pairs2}

    # If we already have all answers by ID, finish
    if len(ans_map) == len(ids_in_order):
        return ans_map

    # --- 2) Full block parsing: ID + YOUR ANSWER inside '=== QUESTION === ... === END ===' ---
    blocks = re.split(r'^\s*=== QUESTION ===\s*$', text, flags=re.MULTILINE)
    for blk in blocks:
        if not blk.strip():
            continue
        blk = re.split(r'^\s*=== END ===\s*$', blk, maxsplit=1, flags=re.MULTILINE)[0]
        m_id = re.search(r'^\s*ID:\s*(.+?)\s*$', blk, flags=re.MULTILINE)
        m_ans = re.search(
            r'^\s*YOUR ANSWER:\s*([ABCD])\s*(?:[).:])?\s*$',
            blk,
            flags=re.IGNORECASE | re.MULTILINE,
        )
        if m_id and m_ans:
            qid = m_id.group(1).strip()
            ans_map[qid] = m_ans.group(1).upper()

    if len(ans_map) == len(ids_in_order):
        return ans_map

    # --- 3) Bare 'YOUR ANSWER:' lines only; map by order ---
    letters = re.findall(
        r'YOUR ANSWER:\s*([ABCD])\s*(?:[).:])?', text, flags=re.IGNORECASE
    )
    letters = [x.upper() for x in letters]

    # Merge: keep any answers we already have by ID; fill gaps by order
    filled = {}
    # First, prefill from ans_map
    for qid in ids_in_order:
        filled[qid] = ans_map.get(qid)

    # Then fill missing from ordered letters
    li = 0
    for qid in ids_in_order:
        if filled[qid] is None:
            if li < len(letters):
                filled[qid] = letters[li]
                li += 1
            else:
                filled[qid] = None

    return filled


def correct_answer_map(question_answers):
    # {id: 'A'|'B'|'C'|'D'}
    return {item['id']: item['correct_answer'].upper() for item in question_answers}


def compare_answers(user_answers, question_answers):
    gt = correct_answer_map(question_answers)
    results = []
    correct = 0
    total = len(gt)
    for qid, true_letter in gt.items():
        user_letter = user_answers.get(qid)
        is_correct = user_letter == true_letter
        results.append(
            {
                'id': qid,
                'llm_answer': user_letter,
                'correct_answer': true_letter,
                'is_correct': is_correct,
            }
        )
        if is_correct:
            correct += 1
    accuracy = correct / total if total else 0.0
    return results, accuracy


story_topics = [
    'sports',
    'technology',
    'medicine',
    'military',
    'family',
    'business',
    'romance',
    'education',
    'churchcrime',
    'travel',
    'philosophy',
    'marriage',
    'friendship',
    'woman',
    'sex',
    'meaning of life',
    'depression',
    'death',
]


def create_story_and_questions(topic: str, num_words, num_questions, model) -> str:
    prompt_text = f"""
    Create a story of {num_words} words.
    The story should be completely fictional and not based on real people or events.
    The topic of the story should be about {topic}.
    Then generate {num_questions} multiple-choice questions with answers.
    The questions should only be answerable with information from the story and not information from somewhere else.

    You must strictly adhere to the following format (do not add or remove tags):

    <story>
    [Insert the story here]
    </story>

    <questions>
        <question id="1">
            <text> [Insert the question text here] </text>
            <answer1> A) [Option text] </answer1>
            <answer2> B) [Option text] </answer2>
            <answer3> C) [Option text] </answer3>
            <answer4> D) [Option text] </answer4>
            <correct_answer> [A) / B) / C) / D)] </correct_answer>
            <source> [Exact line or passage from the story] </source>
        </question>

        <!-- Repeat the same structure for all questions -->
    </questions>
    """

    reply = llm_call(model, prompt_text)
    return str(reply)


def compute_question_index(story_index, number_of_stories):
    if story_index < number_of_stories - 1:
        question_index = story_index + 1
    else:
        question_index = 0
    return question_index


# The story should be about {spec['topic']} and told in a {spec['tone']} {spec['pov']} {spec['shape']} format.


@hydra.main(config_path='.', config_name='config')
def main(cfg: DictConfig):
    assert cfg.num_stories > 1, 'In order to swap stories we need more than 1 story'
    eval_output_dir = Path(
        f'/home/tommaso/repo/OpenHands/bias_check/outputs/{cfg.timestamp.split("_")[0]}'
    )

    metadata_dir = eval_output_dir / get_folder_path_name(cfg) / cfg.model
    metadata_dir.mkdir(parents=True, exist_ok=True)
    Path(metadata_dir / '.hydra').mkdir(parents=True, exist_ok=True)

    for story_index in range(cfg.num_stories):
        rng = random.Random(story_index)
        topic = rng.choice(story_topics)
        story_with_questions = create_story_and_questions(
            topic, cfg.num_words, cfg.num_questions, cfg.generator_model
        )

        with open(metadata_dir / f'story_questions_{story_index}.txt', 'w') as f:
            f.write(story_with_questions)

        # if i == 0:
        #     with open(metadata_dir / 'prompt_generation.txt', 'w') as f:
        #         f.write(story_with_questions)

    # Open all saved txt files
    # metadata_dir = eval_output_dir / get_folder_path_name(cfg) / cfg.model
    files = sorted(metadata_dir.glob('story_questions_*.txt'))

    parsed_files = []  # dict['story','question']
    for file in files:
        with open(file, 'r') as f:
            content = f.read()
            try:
                parsed_content = parse_story_and_questions(content)
            except ET.ParseError as e:
                # E.g. missing / in closing tag
                print(f'Error parsing file {file}: {e}')
                continue
            parsed_files.append(parsed_content)

    performance = []

    for story_index, parsed in enumerate(parsed_files):
        story = parsed['story']

        questions_index = compute_question_index(story_index, len(parsed_files))
        # Get questions from next story
        next_file_content = parsed_files[questions_index]

        # Get all questions, correct answer and source from next_file_content
        question_answers = []
        for item in next_file_content['questions']:
            question_answers.append(
                {
                    'id': item['id'],
                    'text': item['text'],
                    'answers': item['answers'],
                    'source': item['source'],
                    'correct_answer': item['correct_answer'],
                }
            )

        prompt_text = build_prompt(question_answers, story)

        # Save prompt to file
        with open(metadata_dir / f'prompt_{story_index}.txt', 'w') as f:
            f.write(prompt_text)

        reply = llm_call(cfg.examinee_model, prompt_text)
        # Save LLM reply to file
        with open(metadata_dir / f'llm_reply_{story_index}.txt', 'w') as f:
            f.write(reply)

        llm_answers = parse_llm_answers(reply, question_answers)

        results, accuracy = compare_answers(llm_answers, question_answers)

        performance.append(
            {
                'story_index': story_index,
                'questions_index': questions_index,
                'num_questions': len(question_answers),
                'num_correct': sum(1 for r in results if r['is_correct']),
                'accuracy': accuracy,
            }
        )

    df = pd.DataFrame(performance)
    df.to_csv('results.csv')

    print(f'Accuracy: {df["accuracy"].mean()}')

    with open(metadata_dir / '.hydra' / 'config.yaml', 'w') as f:
        OmegaConf.save(cfg, f)


if __name__ == '__main__':
    main()
