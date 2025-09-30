import asyncio
import json
import os
import re
from pathlib import Path
from typing import Any

import hydra
import pandas as pd
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

from evaluation.utils.shared import (
    EvalMetadata,
    EvalOutput,
    codeact_user_response,
    get_default_sandbox_config_for_eval,
    make_metadata,
    reset_logger_for_multiprocessing,
    run_evaluation,
)
from openhands.controller.state.state import State
from openhands.core.config import (
    ExtendedConfig,
    OpenHandsConfig,
    get_llm_config_arg,
)
from openhands.core.logger import openhands_logger as logger
from openhands.core.main import create_runtime, run_controller
from openhands.events.action import CmdRunAction, MessageAction
from openhands.events.action.commands import IPythonRunCellAction
from openhands.events.observation import CmdOutputObservation

# remove when it becomes unnecessary
from openhands.events.serialization.event import event_to_dict
from openhands.runtime.base import Runtime
from openhands.utils.async_utils import call_async_from_sync

from .benchmark_additions import kill_instance, safe_append
from .evaluation import evaluate_model_answer

AGENT_CLS_TO_FAKE_USER_RESPONSE_FN = {
    'CodeActAgentSimulink': codeact_user_response,
}

LOCAL_DATASET_PATH = os.path.join(os.path.dirname(__file__), 'benchmark')


def format_task_dict(example, use_knowledge):
    task = {
        'instance_id': example['instance_id'],
        'task_inst': example['task_inst'],
        'dataset_path': '/benchmark/datasets/'
        + example['dataset_folder_tree'].split('\n')[0][4:],
        'dataset_folder_tree': example['dataset_folder_tree'],
        'dataset_preview': example['dataset_preview'],
        'pred_program_name': 'pred_' + example['gold_program_name'],
    }

    if use_knowledge:
        task['task_inst'] += '\n' + str(example['domain_knowledge'])

    return task


def configure_app_for_evaluation(
    metadata: EvalMetadata, cfg: OmegaConf
) -> OpenHandsConfig:
    sandbox_config = get_default_sandbox_config_for_eval()
    sandbox_config.base_container_image = 'python:3.12-bookworm'
    sandbox_config.runtime_extra_deps = '/openhands/poetry/openhands-ai-5O4_aCHf-py3.12/bin/python  -m pip install numpy matplotlib pandas scikit-learn'
    metadata.agent_class = 'CodeActAgentSimulink'

    runtime_mode = 'docker'
    if not Path('/var/run/docker.sock').exists():
        runtime_mode = 'local'

    config = OpenHandsConfig(
        default_agent=metadata.agent_class,
        run_as_openhands=False,
        runtime=runtime_mode,
        max_budget_per_task=cfg.max_budget_per_task,
        extended=ExtendedConfig({'cfg': cfg}),
        sandbox=sandbox_config,
        # do not mount workspace
        workspace_base=None,
        workspace_mount_path=None,
    )
    config.set_llm_config(metadata.llm_config)
    agent_config = config.get_agent_config(metadata.agent_class)

    agent_config.enable_editor = False
    agent_config.enable_cmd = True
    agent_config.enable_history_truncation = False
    agent_config.enable_prompt_extensions = False
    agent_config.enable_som_visual_browsing = False
    agent_config.enable_browsing = cfg.enable_browsing_for_pictures
    agent_config.enable_cmd = True
    agent_config.enable_think = True
    return config


def initialize_runtime(
    runtime: Runtime,
    instance: pd.Series,  # this argument is not required
    cfg: OmegaConf,
):
    """Initialize the runtime for the agent.

    This function is called before the runtime is used to run the agent.
    """
    logger.info(f'{"-" * 50} BEGIN Runtime Initialization Fn {"-" * 50}')
    obs: CmdOutputObservation

    if instance['class_type'] == 'simulink':
        base_path = Path(
            f'evaluation/benchmarks/simulink/tasks/{cfg.simulation_example}/'
        )
        if cfg.level == 'data_features_diagram':
            path_time_series_data = base_path / 'data.csv'
            runtime.copy_to(path_time_series_data, '/workspace')

            # copy all files from /diagrams
            for file in (base_path / 'diagram').glob('*'):
                # Exclude files with .mat ending
                if file.suffix == '.mat':
                    continue
                runtime.copy_to(file, '/workspace/diagrams')

            logger.info(
                'Level 1 Context: Time series data + Feature names + Simulink diagrams'
            )

        # elif cfg.level == 'data_features_system_description':
        #     path_correct_simulation = base_path / 'correct_simulation.csv'
        #     path_fault_simulation = base_path / 'fault_simulation.csv'
        #     path_description = base_path / 'system_description.txt'
        #     runtime.copy_to(path_correct_simulation, '/workspace')
        #     runtime.copy_to(path_fault_simulation, '/workspace')
        #     runtime.copy_to(path_description, '/workspace')
        #     logger.info(
        #         'Level 2 Context: Time series data + Feature names + (High-level description of the control system)'
        #     )

        # elif cfg.level == 'numerical_data_only':
        #     path_correct_simulation = (
        #         base_path / 'correct_simulation_numerical_data_only.csv'
        #     )
        #     path_fault_simulation = (
        #         base_path / 'fault_simulation_numerical_data_only.csv'
        #     )
        #     runtime.copy_to(path_correct_simulation, '/workspace')
        #     runtime.copy_to(path_fault_simulation, '/workspace')
        #     logger.info('Level 3 Context: Time series data (No feature names)')

    # runtime.copy_to("/home/tommaso/repos/OpenHands/evaluation/benchmarks/ucr_dataset/test.py", '/workspace')
    # Check the database is copied
    action = CmdRunAction(command='cd /workspace && ls -l')
    obs = runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    assert obs.exit_code == 0
    # assert f'time_series_instance.csv' in obs.content
    logger.info(f'{"-" * 50} END Runtime Initialization Fn {"-" * 50}')


def complete_runtime(state: State, metadata_task: json) -> dict[str, Any]:
    """Complete the runtime for the agent.

    This function is called before the runtime is used to run the agent.
    If you need to do something in the sandbox to get the correctness metric after
    the agent has run, modify this function.
    """
    model_answer = {}

    # breakpoint()

    try:
        # Latest message action
        try:
            proposed_solution = state.history[len(state.history) - 1].final_thought
        except:
            proposed_solution = state.history[len(state.history) - 1].content

        match = re.search(r'Final answer:.*$', proposed_solution, re.MULTILINE)
        if match:
            complete_answer = match.group(0)

        model_answer = complete_answer.split('Final answer:')[1].strip()

        # TODO: Include option letter directly in the possible answers and then check for the letter only
        # Remove letter if included
        model_answer = re.sub(r'\b[A-D]\)\s*', '', model_answer)

        # Call script to evaluate the answer
        result = evaluate_model_answer(model_answer, metadata_task)
    except Exception as e:
        logger.error(f'Error during evaluation: {e}')
        result = {}

    return result


def process_instance(
    instance: pd.Series,
    metadata: EvalMetadata,
    reset_logger: bool = True,
    cfg: OmegaConf = None,
) -> EvalOutput:
    instance_id = instance.instance_id  # .replace('/', '__')

    base_path = Path(f'evaluation/benchmarks/simulink/tasks/{cfg.simulation_example}/')

    # Set up the logger properly, so you can run multi-processing to parallelize the evaluation
    if reset_logger:
        log_dir = os.path.join(metadata.eval_output_dir, 'infer_logs')
        reset_logger_for_multiprocessing(logger, instance_id, log_dir)
    else:
        logger.info(f'Starting evaluation for instance {instance_id}.')

    # create the runtime configuration for this evaluation run
    config = configure_app_for_evaluation(metadata, cfg)
    instruction = ''

    if instance['class_type'] == 'simulink':
        # instruction = f"""You are given a data file from a control system. One of them is faulty. Your task is to identify the root cause of the fault. The files are located in /workspace/ Provide your answer in the following form: <sol> Signal: "faulty signal" Timestamp: "time of the fault" </sol> \n.

        # if 'BouncingBall' in cfg.simulation_example:
        # Open metadata.json in base_path and read the content
        with open(base_path / 'metadata_task.json', 'r') as f:
            metadata_task = json.load(f)

        # Format choices A), B), ...
        # instruction = f"""{metadata_task['question']} \n {'\n'.join(f'{chr(65 + i)}) {msg}' for i, msg in enumerate(metadata_task['options']))}"""
        # TODO: Put improved instruction in the metadata
        improved_prompt = "You are given a simulation and your task is to determine whether any physically implausible events occur at any time point. Select the correct answer in the list. Note that some answers also require you to return the time something happened"
        instruction = f"""{improved_prompt} \n {'\n'.join(f'{chr(65 + i)}) {msg}' for i, msg in enumerate(metadata_task['options']))}"""

        instruction += (
            '\n Please provide your response in the following format:\n'
            'Final answer: <selected option text with the missing information filled inside the curly brackets {}> Do not remove the curly brackets {}.\n'
        )

        if cfg.level == 'data_features_diagram':
            # instruction += 'The correct simulation data is in correct_simulation.csv, the faulty simulation data is in fault_simulation.csv and the diagram of the control system is in diagram.png.'
            instruction += (
                ' Diagrams of the control system are located in /workspace/diagrams.'
            )

        elif cfg.level == 'data_features_system_description':
            instruction += 'The correct simulation data is in correct_simulation.csv, the faulty simulation data is in fault_simulation.csv and a high-level description of the control system is in system_description.txt.'
        elif cfg.level == 'numerical_data_only':
            instruction += 'The correct simulation data is in correct_simulation_numerical_data_only.csv and the faulty simulation data is in fault_simulation_numerical_data_only.csv.'

        # Level 1: Time series data + Feature names + Simulink diagram
        # Level 2: Time series data + Feature names + (High-level description of the control system)
        # Level 3: Time series data

    if cfg.sid:
        sid = cfg.sid
    else:
        sid = None

    # Overwrite max_interactions
    config.max_iterations = cfg.max_iterations
    runtime = create_runtime(config, sid=sid)

    call_async_from_sync(runtime.connect)
    initialize_runtime(runtime, instance, cfg=cfg)
    # Here's how you can run the agent (similar to the `main` function) and get the final task state

    state: State | None = asyncio.run(
        run_controller(
            config=config,
            initial_user_action=MessageAction(content=instruction),
            runtime=runtime,
            fake_user_response_fn=AGENT_CLS_TO_FAKE_USER_RESPONSE_FN.get(
                metadata.agent_class
            ),
        )
    )

    # AD: What is this?
    [code.code for code in state.history if isinstance(code, IPythonRunCellAction)]

    # Open metadata_task.json in base_path and read the content
    with open(base_path / 'metadata_task.json', 'r') as f:
        metadata_task = json.load(f)

    # ======= Attempt to evaluate the agent's edits =======
    evaluation_result = complete_runtime(state, metadata_task)

    # Save the evaluation result
    eval_output_dir = Path(
        f'evaluation/evaluation_outputs/outputs/{cfg.timestamp.split("_")[0]}'
    )
    results_dir = eval_output_dir / get_folder_path_name(cfg)

    with open(results_dir / 'results.json', 'w') as f:
        json.dump(evaluation_result, f)

    # If you are working on some simpler benchmark that only evaluates the final model output (e.g., in a MessageAction)
    # You can simply get the LAST `MessageAction` from the returned `state.history` and parse it for evaluation.
    if state is None:
        raise ValueError('State should not be None.')
    metrics = state.metrics.get() if state.metrics else None

    # history is now available as a stream of events, rather than list of pairs of (Action, Observation)
    # for compatibility with the existing output format, we can remake the pairs here

    # AD: Still needed?
    for x in state.history:
        if 'content' in x.__dict__:
            text = x.content
            # replace base64 images with a placeholder
            splitted = text.split('\n')
            for i, line in enumerate(splitted):
                if '![image](data:image/png;base64,' in line:
                    # breakpoint()
                    # with open(current / f'{i}.png', 'wb') as f:
                    #     f.write(png.encode('utf-8'))

                    splitted[i] = (
                        '![image](data:image/png;base64, ...) already displayed to user'
                    )
            text = '\n'.join(splitted)
            x.content = text
    histories = [event_to_dict(x) for x in state.history]

    # Save the output
    output = EvalOutput(
        instance_id=str(instance.instance_id),
        instruction=instruction,
        metadata=metadata,
        history=histories,
        metrics=metrics,
        error=state.last_error if state and state.last_error else None,
        test_result=evaluation_result,
        sid=runtime.sid,
    )

    return output


def get_folder_path_name(cfg: OmegaConf):
    try:
        run_idx = HydraConfig.get().job.num
    except Exception:
        run_idx = int(os.environ.get('HYDRA_JOB_NUM', 0))

    return cfg.timestamp + '_' + str(run_idx)


def prepare_evaluation(
    cfg: OmegaConf = None,
):
    keys = [
        {'class_type': cfg.class_type, 'example': cfg.instance, 'fold': cfg.fold}
        for i in range(cfg.number_of_experiments)
    ]
    dataset = pd.DataFrame(keys)

    # Create instance_id (hydra date and time)
    # instance_id = get_folder_path_name(cfg).split('_')[0]
    instance_id = get_folder_path_name(cfg)
    dataset['instance_id'] = instance_id

    return dataset


@hydra.main(config_path='hydra_config', config_name='main')
def main(cfg):
    args = cfg

    llm_config = None
    if args.llm_config:
        llm_config = get_llm_config_arg(args.llm_config)
        # modify_params must be False for evaluation purpose, for reproducibility and accurancy of results
        if llm_config is not None:
            llm_config.modify_params = False
    if llm_config is None:
        raise ValueError(f'Could not find LLM config: --llm_config {args.llm_config}')

    eval_output_dir = Path(
        f'evaluation/evaluation_outputs/outputs/{cfg.timestamp.split("_")[0]}'
    )
    metadata_dir = eval_output_dir / get_folder_path_name(cfg)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    # Save the hydra config file inside the metadata dir
    from omegaconf import OmegaConf

    # save cfg
    Path(metadata_dir / '.hydra').mkdir(parents=True, exist_ok=True)
    with open(metadata_dir / '.hydra' / 'config.yaml', 'w') as f:
        OmegaConf.save(cfg, f)
    args.max_iterations = cfg.max_iterations
    args.eval_output_dir = str(metadata_dir)
    metadata = make_metadata(
        llm_config,
        str(cfg.instance),
        args.agent_cls,
        args.max_iterations,
        args.eval_note,
        args.eval_output_dir,
    )

    # Open metadata_task.json in base_path and read the content
    base_path = Path(f'evaluation/benchmarks/simulink/tasks/{cfg.simulation_example}/')
    with open(base_path / 'metadata_task.json', 'r') as f:
        metadata_task = json.load(f)

    # Append metadata_task to metadata
    # mode="json" makes sure SecretStr and other exotic types are converted into strings
    metadata_json = metadata.model_dump(mode="json")
    metadata_json['metadata_task'] = metadata_task

    # Save the metadata to a json file in the eval_output_dir again (make_metadata was called before)
    metadata_file_path = os.path.join(metadata.eval_output_dir, 'metadata.json')
    with open(metadata_file_path, 'w') as f:
        json.dump(metadata_json, f)

    output_file = os.path.join(metadata.eval_output_dir, 'output.jsonl')

    instances = prepare_evaluation(cfg)
    repetition_per_instance = cfg.number_of_experiments
    instances = pd.concat([instances] * repetition_per_instance, ignore_index=True)
    run_evaluation(
        instances,
        metadata,
        output_file,
        args.eval_num_workers,
        process_instance,
        process_instance_kwargs={'cfg': cfg},
        max_retries=cfg.max_retries,
    )

    # Add the output file to the trajectory visualiser folder

    # Check if the trajectory visualiser folder exists
    if not Path(cfg.trajectory_visualiser_folder).exists():
        # Log a warning if the folder does not exist
        logger.warning(
            f'Trajectory visualiser folder {cfg.trajectory_visualiser_folder} does not exist. Creating it.'
        )
        # Create the trajectory visualiser folder if it does not exist
        Path(cfg.trajectory_visualiser_folder).mkdir(parents=True, exist_ok=True)
    target_path = Path(cfg.trajectory_visualiser_folder) / 'output.jsonl'
    if not target_path.exists():
        # Create an empty file if it doesn't exist
        target_path.touch()

    # Open the output file file
    with open(output_file, 'r') as f:
        # Read the content of the file
        content = f.read()

    safe_append(path=target_path, text=content)
    # Open the output file and read the sid
    kill_instance(output_file)

    # os.system('docker kill $(docker ps -q)')


if __name__ == '__main__':
    main()
