import os
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
from openai import OpenAI


def get_folder_path_name(cfg: OmegaConf):
    try:
        run_idx = HydraConfig.get().job.num
    except Exception:
        run_idx = int(os.environ.get('HYDRA_JOB_NUM', 0))

    return cfg.timestamp + '_' + str(run_idx)


def model_client(model_name: str):
    if 'gpt' in model_name:
        client = OpenAI()
    return client


@hydra.main(config_path='.', config_name='config')
def main(cfg: DictConfig):
    eval_output_dir = Path(
        f'/home/tommaso/repo/OpenHands/task_questions/outputs/{cfg.timestamp.split("_")[0]}'
    )

    basic_instruction = f"""You are given the reasoning behind a fault introduced in a control system.
    The control system is called {cfg.task_name} and developed in Simulink. You need to generate questions that test understanding of this fault.
    The question should be clear, concise, and relevant to the fault described. Create {cfg.num_questions} questions and list them numerically.
    Your answer should only contain the questions."""

    prompt = f'{basic_instruction}\n\nFault description: {cfg.fault_description}'
    cfg.prompt = prompt

    client = model_client(cfg.openai.model)

    response = client.responses.create(
        model=cfg.openai.model,
        input=[
            {
                'role': 'developer',
                'content': 'You are an expert on control systems.',
            },
            {'role': 'user', 'content': prompt},
        ],
        # temperature=cfg.openai.temperature,
    )

    reply = response.output_text

    metadata_dir = eval_output_dir / get_folder_path_name(cfg) / cfg.task_name
    metadata_dir.mkdir(parents=True, exist_ok=True)

    Path(metadata_dir / '.hydra').mkdir(parents=True, exist_ok=True)
    with open(metadata_dir / '.hydra' / 'config.yaml', 'w') as f:
        OmegaConf.save(cfg, f)

    with open(metadata_dir / 'generated_questions.txt', 'w') as f:
        f.write(reply)


if __name__ == '__main__':
    main()
