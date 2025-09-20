from openai import OpenAI
import hydra
from omegaconf import OmegaConf
from omegaconf import DictConfig
from pathlib import Path
import os
import re
import xml.etree.ElementTree as ET
from typing import List, Dict
import anthropic
import random


def get_folder_path_name(cfg: OmegaConf):
    try:
        run_idx = HydraConfig.get().job.num
    except Exception:
        run_idx = int(os.environ.get('HYDRA_JOB_NUM', 0))

    return cfg.timestamp + '_' + str(run_idx)


def model_client(model_name):
    if "gpt-5" == model_name:
        client = OpenAI(base_url="https://api.openai.com/v1")
    elif "claude-sonnet-4-20250514" == model_name:
        client = anthropic.Anthropic()
    else:
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ.get("OPENROUTER_API_KEY"),
        )

    return client


def _clean_text(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def _normalize_correct_answer(s: str) -> str:
    """
    Extract a single letter A-D from values like:
    'A', 'A)', '[A)', 'Correct: A)', etc.
    Falls back to cleaned raw text if no A-D found.
    """
    s = _clean_text(s)
    m = re.search(r"\b([A-Da-d])\s*[\)\].]?\b", s)
    return m.group(1).upper() if m else s


def parse_story_and_questions(raw: str) -> Dict:
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
    wrapped = f"<root>{raw}</root>"
    root = ET.fromstring(wrapped)

    # Story
    story_el = root.find("story")
    story = _clean_text(story_el.text if story_el is not None else "")

    # Questions
    questions_el = root.find("questions")
    questions: List[Dict] = []
    if questions_el is not None:
        for q in questions_el.findall("question"):
            qid = q.attrib.get("id", "")

            # Question text
            qtext_el = q.find("text")
            qtext = _clean_text(qtext_el.text if qtext_el is not None else "")

            # Answers in order answer1..answer4
            answers = []
            for i in range(1, 5):
                a_el = q.find(f"answer{i}")
                if a_el is not None and (a_el.text or "").strip():
                    answers.append(_clean_text(a_el.text))

            # Correct answer (normalize to A-D)
            ca_el = q.find("correct_answer")
            correct_answer = _normalize_correct_answer(
                ca_el.text if ca_el is not None else ""
            )

            # Source
            src_el = q.find("source")
            source = _clean_text(src_el.text if src_el is not None else "")

            questions.append(
                {
                    "id": qid,
                    "text": qtext,
                    "answers": answers,
                    "correct_answer": correct_answer,
                    "source": source,
                }
            )

    return {"story": story, "questions": questions}


# def build_prompt(question_answers, story):
#     blocks = []

#     # Story section at the top
#     blocks.append("=== STORY ===")
#     blocks.append(story.strip())
#     blocks.append("=== END STORY ===\n")

#     # Questions
#     for item in question_answers:
#         q_lines = []
#         q_lines.append("=== QUESTION ===")
#         q_lines.append(f"ID: {item['id']}")
#         q_lines.append(f"TEXT: {item['text']}")

#         # Options
#         labels = ["A", "B", "C", "D"]
#         for lab, opt in zip(labels, item["answers"][:4]):
#             # Optional: strip leading "A) " etc.
#             cleaned = opt.strip()
#             if cleaned.startswith(f"{lab}) "):
#                 cleaned = cleaned[len(f"{lab}) ") :]
#             q_lines.append(f"OPTION {lab}: {cleaned}")

#         q_lines.append("YOUR ANSWER: ")  # placeholder for LLM
#         q_lines.append("=== END ===")
#         blocks.append("\n".join(q_lines))

#     instructions = (
#         "You are given a story and multiple-choice questions about it.\n"
#         "Fill in ONLY the 'YOUR ANSWER:' lines with a single letter A, B, C, or D.\n"
#         "Return the ENTIRE prompt with the answers filled in, keeping the same structure."
#         "Do NOT change anything else."
#         "If you don't know the answer, nevertheless pick one of A, B, C, or D."
#     )

#     return instructions + "\n\n" + "\n\n".join(blocks)


def build_prompt(question_answers, story):
    blocks = []

    # Instructions: compact answers only
    instructions = (
        "You are given a story and multiple-choice questions about it.\n"
        "Return ONLY this block with one line per question as <id>=<A/B/C/D>:\n"
        "<answers>\n"
        "1=A\n"
        "2=B\n"
        "</answers>\n"
        "Do NOT change anything else."
        "If you don't know the answer, nevertheless pick one of A, B, C, or D."
    )

    # Story section at the top
    blocks.append("=== STORY ===")
    blocks.append(story.strip())
    blocks.append("=== END STORY ===\n")

    # Questions (keep as-is for context)
    for item in question_answers:
        q_lines = []
        q_lines.append("=== QUESTION ===")
        q_lines.append(f"ID: {item['id']}")
        q_lines.append(f"TEXT: {item['text']}")

        labels = ["A", "B", "C", "D"]
        for lab, opt in zip(labels, item["answers"][:4]):
            cleaned = opt.strip()
            if cleaned.startswith(f"{lab}) "):
                cleaned = cleaned[len(f"{lab}) ") :]
            q_lines.append(f"OPTION {lab}: {cleaned}")

        # No placeholder line (saves tokens in the model's output)
        q_lines.append("=== END ===")
        blocks.append("\n".join(q_lines))

    return instructions + "\n\n" + "\n\n".join(blocks)


# def parse_llm_answers(output_text):
#     """
#     Returns dict: {id_str: 'A'|'B'|'C'|'D'}
#     Robust to extra whitespace and multi-line TEXT.
#     """
#     answers = {}
#     # Split into blocks
#     blocks = re.split(r"^\s*=== QUESTION ===\s*$", output_text, flags=re.MULTILINE)
#     for block in blocks:
#         if not block.strip():
#             continue
#         # Ensure this block ends before the next by cutting at === END ===
#         block = re.split(r"^\s*=== END ===\s*$", block, maxsplit=1, flags=re.MULTILINE)[
#             0
#         ]

#         # ID
#         m_id = re.search(r"^\s*ID:\s*(.+?)\s*$", block, flags=re.MULTILINE)
#         if not m_id:
#             continue
#         qid = m_id.group(1).strip()

#         # YOUR ANSWER
#         m_ans = re.search(
#             r"^\s*YOUR ANSWER:\s*([ABCD])\s*$",
#             block,
#             flags=re.MULTILINE | re.IGNORECASE,
#         )
#         if not m_ans:
#             continue
#         ans = m_ans.group(1).upper()

#         answers[qid] = ans
#     return answers


# def parse_llm_answers(output_text: str, question_answers: List[dict]) -> Dict[str, str]:
#     """
#     Parse LLM output to a dict {question_id: 'A'|'B'|'C'|'D'}.
#     Handles:
#       - Full structured blocks with IDs and YOUR ANSWER lines.
#       - Degraded outputs with only repeated YOUR ANSWER lines.

#     Strategy:
#       1) Try to parse per-question blocks and capture both ID and answer in the same block.
#       2) If that yields an incomplete set, fall back to order-based extraction of all YOUR ANSWER letters.

#     If there are fewer answers than questions, missing ones map to None.
#     Extra answers are ignored.
#     """
#     # Normalize newlines (helpful if CRLF sneaks in)
#     text = output_text.replace("\r\n", "\n").replace("\r", "\n")

#     # --- Primary: parse block-wise with ID + YOUR ANSWER ---
#     ans_map = {}

#     # Split into possible blocks after '=== QUESTION ==='
#     blocks = re.split(r"^\s*=== QUESTION ===\s*$", text, flags=re.MULTILINE)
#     for blk in blocks:
#         if not blk.strip():
#             continue
#         # Cut each block at the first END marker
#         blk = re.split(r"^\s*=== END ===\s*$", blk, maxsplit=1, flags=re.MULTILINE)[0]

#         # Find ID and YOUR ANSWER within the same block
#         m_id = re.search(r"^\s*ID:\s*(.+?)\s*$", blk, flags=re.MULTILINE)
#         m_ans = re.search(
#             r"^\s*YOUR ANSWER:\s*([ABCD])\s*(?:[).:])?\s*$",
#             blk,
#             flags=re.IGNORECASE | re.MULTILINE,
#         )
#         if m_id and m_ans:
#             qid = m_id.group(1).strip()
#             letter = m_ans.group(1).upper()
#             ans_map[qid] = letter

#     # If we got a full set (every question has an answer), return it
#     ids_in_order = [q["id"] for q in question_answers]
#     if len(ans_map) == len(ids_in_order):
#         return ans_map

#     # --- Fallback: order-based extraction of YOUR ANSWER lines only ---
#     letters = re.findall(
#         r"YOUR ANSWER:\s*([ABCD])\s*(?:[).:])?", text, flags=re.IGNORECASE
#     )
#     letters = [x.upper() for x in letters]

#     fallback = {}
#     for i, qid in enumerate(ids_in_order):
#         fallback[qid] = letters[i] if i < len(letters) else None

#     return fallback


def parse_llm_answers(output_text: str, question_answers: List[dict]) -> Dict[str, str]:
    """
    Returns {question_id: 'A'|'B'|'C'|'D' or None}.
    Priority:
      1) Compact <answers> block with lines: "<id>=<A/B/C/D>"
      2) Full blocks with ID + 'YOUR ANSWER: X'
      3) Bare 'YOUR ANSWER: X' lines mapped by order
    """
    text = output_text.replace("\r\n", "\n").replace("\r", "\n")
    ids_in_order = [q["id"] for q in question_answers]

    # --- 1) Compact <answers> block ---
    # Try to isolate the block; if not present, consider the whole text.
    m = re.search(
        r"<answers>\s*(.*?)\s*</answers>", text, flags=re.DOTALL | re.IGNORECASE
    )
    block = m.group(1) if m else text

    # Pattern: "id = Letter"
    pairs = re.findall(
        r"^\s*([0-9A-Za-z_-]+)\s*=\s*([ABCD])\s*$",
        block,
        flags=re.MULTILINE | re.IGNORECASE,
    )
    ans_map = {qid: ans.upper() for qid, ans in pairs}

    # Also accept verbose lines like "Question ID: 2: Answer: A"
    if not ans_map:
        pairs2 = re.findall(
            r"^\s*(?:Question\s*ID:\s*)?([0-9A-Za-z_-]+)\s*[:=]\s*(?:Answer:\s*)?([ABCD])\s*$",
            block,
            flags=re.MULTILINE | re.IGNORECASE,
        )
        ans_map = {qid: ans.upper() for qid, ans in pairs2}

    # If we already have all answers by ID, finish
    if len(ans_map) == len(ids_in_order):
        return ans_map

    # --- 2) Full block parsing: ID + YOUR ANSWER inside '=== QUESTION === ... === END ===' ---
    blocks = re.split(r"^\s*=== QUESTION ===\s*$", text, flags=re.MULTILINE)
    for blk in blocks:
        if not blk.strip():
            continue
        blk = re.split(r"^\s*=== END ===\s*$", blk, maxsplit=1, flags=re.MULTILINE)[0]
        m_id = re.search(r"^\s*ID:\s*(.+?)\s*$", blk, flags=re.MULTILINE)
        m_ans = re.search(
            r"^\s*YOUR ANSWER:\s*([ABCD])\s*(?:[).:])?\s*$",
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
        r"YOUR ANSWER:\s*([ABCD])\s*(?:[).:])?", text, flags=re.IGNORECASE
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
    return {item["id"]: item["correct_answer"].upper() for item in question_answers}


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
                "id": qid,
                "llm_answer": user_letter,
                "correct_answer": true_letter,
                "is_correct": is_correct,
            }
        )
        if is_correct:
            correct += 1
    accuracy = correct / total if total else 0.0
    return results, accuracy


TOPICS = [
    "Night of Lanterns festival",
    "Pneumatic post network",
    "Traveler's pocket almanac (13-month calendar)",
    "Museum placards for imaginary exhibits",
    "Hex-tile alley game with glyph scoring",
    "Bioluminescent forest with pulse cycles",
    "School timetable with rotating blocks",
    "Glacier dig field notes",
    "Teahouse menu with house measures",
    "Ring-city tram network",
    "Lighthouse keeper's log",
    "Emotion-indexed library",
    "Constellation ritual calendar",
    "Airship cargo manifest with sigils",
    "Clockwork animal sanctuary",
    "Festival program with token currency",
    "Submersible dive log",
    "Guild price list (apprentice/maker/master)",
    "Skyport docking chart",
    "Desert burrower scent-markers",
    "Court of minor disputes with seed stones",
    "Census of imaginary professions",
    "Dream market leasing rules",
    "River pact knot-ribbons",
]

MOD_TONE = [
    "whimsical",
    "matter-of-fact",
    "mystical",
    "dry administrative",
    "reportage",
    "epistolary",
]
MOD_POV = [
    "first-person",
    "second-person",
    "third-person limited",
    "omniscient",
    "documentary",
]
MOD_SHAPE = [
    "log entries",
    "notices",
    "manual",
    "menu",
    "itinerary",
    "ledger",
    "field report",
    "FAQ",
]


story_topics = [
    "sports",
    "technology",
    "medicine",
    "military",
    "family",
    "business",
    "romance",
    "education",
    "church"
    "crime",
    "travel",
    "philosophy",
    "marriage",
    "friendship",
    "woman",
    "sex",
    "meaning of life",
    "depression",
    "death",
]


def sample_topic(rng: random.Random):
    topic = rng.choice(TOPICS)
    tone = rng.choice(MOD_TONE)
    pov = rng.choice(MOD_POV)
    shape = rng.choice(MOD_SHAPE)
    return {"topic": topic, "tone": tone, "pov": pov, "shape": shape}

# The story should be about {spec['topic']} and told in a {spec['tone']} {spec['pov']} {spec['shape']} format.


@hydra.main(config_path=".", config_name="config")
def main(cfg: DictConfig):

    eval_output_dir = Path(
        f'/home/tommaso/repo/OpenHands/bias_check/outputs/{cfg.timestamp.split("_")[0]}'
    )

    for i in range(cfg.num_stories):
        rng = random.Random()  # or seed per run for reproducibility
        topic = rng.choice(story_topics)

        prompt = f"""
        Create a story of {cfg.num_words} words.
        The story should be completely fictional and not based on real people or events.
        The topic of the story should be about {topic}.
        Then generate {cfg.num_questions} multiple-choice questions with answers.
        The questions should only be answerable with information from the story and not information from somwhere else.

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

        if "gpt-5" == cfg.model:
            breakpoint()
            client = model_client(cfg.model)
            response = client.responses.create(
                model=cfg.model,
                input=[
                    {
                        "role": "developer",
                        "content": "You are a helpful AI assistant that doesn't make mistakes.",
                    },
                    {"role": "user", "content": prompt},
                ],
            )
            reply = response.output_text

        else:
            client = model_client(cfg.model)
            response = client.chat.completions.create(
                model=cfg.model,
                messages=[
                    {
                        "role": "developer",
                        "content": "You are a helpful AI assistant that doesn't make mistakes.",
                    },
                    {"role": "user", "content": prompt},
                ],
            )
            reply = response.choices[0].message.content

        metadata_dir = eval_output_dir / get_folder_path_name(cfg) / cfg.model
        metadata_dir.mkdir(parents=True, exist_ok=True)

        with open(metadata_dir / f'story_questions_{i}.txt', 'w') as f:
            f.write(reply)

        if i == 0:
            with open(metadata_dir / f'prompt_generation.txt', 'w') as f:
                f.write(prompt)

    # Open all saved txt files
    metadata_dir = eval_output_dir / get_folder_path_name(cfg) / cfg.model
    files = sorted(metadata_dir.glob("story_questions_*.txt"))

    parsed_files = []
    for file in files:
        with open(file, 'r') as f:
            content = f.read()
            try:
                parsed_content = parse_story_and_questions(content)
            except ET.ParseError as e:
                # E.g. missing / in closing tag
                print(f"Error parsing file {file}: {e}")
                continue
            parsed_files.append(parsed_content)

    performance = []

    for i, parsed in enumerate(parsed_files):
        story = parsed["story"]

        # Get questions from next story
        if i < len(parsed_files) - 1:
            next_file_content = parsed_files[i + 1]
        else:
            next_file_content = parsed_files[0]

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
        with open(metadata_dir / f'prompt_{i}.txt', 'w') as f:
            f.write(prompt_text)

        if "gpt-5" == cfg.model:
            client = model_client(cfg.model)
            response = client.responses.create(
                model=cfg.model,
                input=[
                    {
                        "role": "developer",
                        "content": "You are a helpful AI assistant that doesn't make mistakes.",
                    },
                    {"role": "user", "content": prompt_text},
                ],
            )
            reply = response.output_text

        else:
            client = model_client(cfg.model)
            response = client.chat.completions.create(
                model=cfg.model,
                messages=[
                    {
                        "role": "developer",
                        "content": "You are a helpful AI assistant that doesn't make mistakes.",
                    },
                    {"role": "user", "content": prompt_text},
                ],
            )
            reply = response.choices[0].message.content

        # Save LLM reply to file
        with open(metadata_dir / f'llm_reply_{i}.txt', 'w') as f:
            f.write(reply)

        llm_answers = parse_llm_answers(reply, question_answers)

        results, accuracy = compare_answers(llm_answers, question_answers)

        performance.append(
            {
                'story_index': i,
                'num_questions': len(question_answers),
                'num_correct': sum(1 for r in results if r['is_correct']),
                'accuracy': accuracy,
            }
        )

    # Average accuracy across all stories
    avg_accuracy = (
        sum(p['accuracy'] for p in performance) / len(performance)
        if performance
        else 0.0
    )
    # Add avg accuracy to perfomrnace dict once
    performance.append({'average_accuracy': avg_accuracy})

    # Save performance as json file
    import json

    with open(metadata_dir / 'performance.json', 'w') as f:
        json.dump(performance, f, indent=4)

    Path(metadata_dir / '.hydra').mkdir(parents=True, exist_ok=True)
    with open(metadata_dir / '.hydra' / 'config.yaml', 'w') as f:
        OmegaConf.save(cfg, f)


if __name__ == "__main__":
    main()
