# /workspace/OpenHands/call_error_agent_completion.py

import argparse
import os
import sys

import toml
from omegaconf import OmegaConf
from unittest.mock import MagicMock

try:
    from openhands.agenthub.error_bench_agent.error_agent import ErrorAgent
    from openhands.core.config import AgentConfig, LLMConfig
    from openhands.core.logger import openhands_logger as logger
    from openhands.llm.llm import LLM
    from openhands.controller.state.state import State
    from openhands.core.message import Message, TextContent
    from openhands.events.action import MessageAction
    from openhands.events.event import EventSource
except ImportError as e:
    print(f'Error importing OpenHands modules: {e}\nPYTHONPATH: {sys.path}')
    sys.exit(1)

# Open hydra


def main():
    parser = argparse.ArgumentParser(
        description='Test ErrorAgent LLM completion using specific LLM config from TOML, with Gemini Pro fallback.'
    )
    parser.add_argument(
        '--config_file',
        type=str,
        default='config.toml',
        help='Path to the TOML configuration file. Default: ./config.toml',
    )
    parser.add_argument(
        '--llm_name',
        type=str,
        default=None,
        help='Name of the LLM configuration (from [[llms]] list in config.toml) to use.',
    )
    parser.add_argument(  # RESTORED
        '--temperature_override',
        type=float,
        default=None,
        help='Override temperature for the selected LLM configuration.',
    )
    parser.add_argument(
        '--hydra_config_file',
        type=str,
        default=None,
        help='Path to the Hydra configuration file. Default: None',
    )

    args = parser.parse_args()
    logger.info(
        f"Starting script. Config File: '{args.config_file}', LLM Name: '{args.llm_name if args.llm_name else '(default handling)'}', Temp Override: {args.temperature_override}"
    )

    selected_llm_config_dict = None
    full_toml_data = None

    # Load Hydra config if specified
    cfg = None
    if args.hydra_config_file:
        if os.path.exists(args.hydra_config_file):
            logger.info(f'Loading Hydra configuration from: {args.hydra_config_file}')
            try:
                cfg = OmegaConf.load(args.hydra_config_file)
            except Exception as e:
                logger.error(f'Failed to load Hydra config file: {e}', exc_info=True)
                return
        else:
            logger.warning(f"Hydra configuration file '{args.hydra_config_file}' not found.")

    GEMINI_PRO_DEFAULTS = {
        'model': 'gemini-1.5-flash-latest',
        'temperature': 0.1,
        'custom_llm_provider': 'gemini',
    }

    selected_llm_config_dict = None
    full_toml_data = None

    config_file_exists = os.path.exists(args.config_file)
    if config_file_exists:
        logger.info(f'Loading configuration from: {args.config_file}')
        full_toml_data = toml.load(args.config_file)
    else:
        logger.warning(f"Configuration file '{args.config_file}' not found.")

    if args.llm_name and config_file_exists and full_toml_data:
        logger.info(f"Attempting to use named LLM configuration: '{args.llm_name}'")
        found_named_config = None
        if 'llms' in full_toml_data and isinstance(full_toml_data['llms'], list):
            for llm_entry in full_toml_data['llms']:
                if (
                    isinstance(llm_entry, dict)
                    and llm_entry.get('name') == args.llm_name
                ):
                    found_named_config = llm_entry.copy()
                    break
        if found_named_config:
            selected_llm_config_dict = found_named_config
            logger.info(
                f"Successfully loaded named LLM configuration '{args.llm_name}'."
            )
        else:
            logger.warning(
                f"LLM configuration named '{args.llm_name}' not found in '{args.config_file}'."
            )

    if (
        selected_llm_config_dict is None
        and config_file_exists
        and full_toml_data
        and not args.llm_name
    ):
        logger.info(
            'No specific LLM name provided by user, attempting to load default [llm] section.'
        )
        default_llm_section = full_toml_data.get('llm', {}).copy()
        if default_llm_section.get('model'):
            selected_llm_config_dict = default_llm_section
            logger.info('Successfully loaded default [llm] section from config file.')
        else:
            logger.warning(
                "Default [llm] section in config file is missing or does not specify a 'model'."
            )

    if selected_llm_config_dict is None:
        logger.warning(
            'No suitable configuration found in TOML or no TOML file. Using Gemini Pro defaults as fallback.'
        )
        selected_llm_config_dict = GEMINI_PRO_DEFAULTS.copy()
        selected_llm_config_dict['model'] = os.getenv(
            'OPENHANDS_LLM_MODEL', selected_llm_config_dict['model']
        )
        selected_llm_config_dict['temperature'] = float(
            os.getenv(
                'OPENHANDS_LLM_TEMPERATURE', selected_llm_config_dict['temperature']
            )
        )

    if (
        args.temperature_override is not None
    ):  # This uses the now-defined args.temperature_override
        selected_llm_config_dict['temperature'] = args.temperature_override
        logger.info(f'Overriding temperature to: {args.temperature_override}')

    if not selected_llm_config_dict or not selected_llm_config_dict.get('model'):
        logger.critical("LLM 'model' could not be determined. Cannot proceed.")
        return

    selected_llm_config_dict['temperature'] = float(
        selected_llm_config_dict.get('temperature', 0.1)
    )

    final_llm_args_for_config_class = {}
    if 'model' in selected_llm_config_dict:
        final_llm_args_for_config_class['model'] = str(
            selected_llm_config_dict['model']
        )
    if 'temperature' in selected_llm_config_dict:
        final_llm_args_for_config_class['temperature'] = float(
            selected_llm_config_dict['temperature']
        )

    if 'custom_llm_provider' in selected_llm_config_dict:
        final_llm_args_for_config_class['custom_llm_provider'] = selected_llm_config_dict['custom_llm_provider']
        logger.info(
            f"Including 'custom_llm_provider' ('{selected_llm_config_dict['custom_llm_provider']}') in LLMConfig."
        )

    llm_config = LLMConfig(**final_llm_args_for_config_class)
    logger.info(
        f"LLMConfig created: model='{llm_config.model}', temperature='{getattr(llm_config, 'temperature', 'N/A')}'"
    )

    # Create a mock LLM object to avoid real API calls
    mock_llm = MagicMock()

    # Configure the mock LLM's completion method to return a simulated response
    # Configure the mock LLM's completion method to return a simulated response
    # Mimic the structure of litellm.types.utils.ModelResponse
    mock_response = MagicMock(spec=['choices', 'id']) # Add 'id' to the spec
    mock_response.id = "mock_response_id_123" # Assign a dummy ID
    mock_response.choices = [MagicMock(spec=['message'])]
    mock_response.choices[0].message = MagicMock(spec=['content', 'tool_calls'])
    mock_response.choices[0].message.content = "A large language model is an AI model trained on a massive amount of text data to understand and generate human-like text."
    mock_response.choices[0].message.tool_calls = None # Ensure no tool calls are returned

    mock_llm.completion.return_value = mock_response

    llm = mock_llm # Use the mock LLM

    logger.info("Mock LLM initialized with structured response.")
# Debugging prints
    print(f"Debug: mock_response type: {type(mock_response)}")
    print(f"Debug: mock_response has choices: {'choices' in dir(mock_response)}")
    if hasattr(mock_response, 'choices'):
        print(f"Debug: mock_response.choices type: {type(mock_response.choices)}")
        print(f"Debug: len(mock_response.choices): {len(mock_response.choices)}")
        if len(mock_response.choices) > 0:
            print(f"Debug: mock_response.choices[0] type: {type(mock_response.choices[0])}")
            print(f"Debug: mock_response.choices[0] has message: {'message' in dir(mock_response.choices[0])}")
            if hasattr(mock_response.choices[0], 'message'):
                print(f"Debug: mock_response.choices[0].message type: {type(mock_response.choices[0].message)}")
                print(f"Debug: mock_response.choices[0].message has content: {'content' in dir(mock_response.choices[0].message)}")
                print(f"Debug: mock_response.choices[0].message has tool_calls: {'tool_calls' in dir(mock_response.choices[0].message)}")

    

    # MODIFIED AgentConfig initialization
    agent_config = AgentConfig()
    logger.info('AgentConfig initialized (default).')

    try:
        error_agent_instance = ErrorAgent(llm=llm, config=agent_config, cfg=cfg)
        logger.info('ErrorAgent instantiated successfully.')
    except Exception as e:
        logger.error(f'Failed to instantiate ErrorAgent: {e}', exc_info=True)

    prompt = 'What is a large language model? Explain in one sentence.'
    logger.info(f"Simulating user message with prompt: '{prompt}'")

    # Create a State object and add the user message to history
    state = State()
    # Create a MessageAction and add it to history
    # The 'source' attribute is added by the State/View logic, not the constructor
    user_message_action = MessageAction(content=prompt)
    state.history.append(user_message_action)

    try:
        # Call the agent's step method to get an action
        action = error_agent_instance.step(state)

        logger.info(f'Agent returned action: {type(action).__name__}')

        # Check if the action is a MessageAction and print its content
        if isinstance(action, Message):
            print(f'\nLLM Output:\n{action.content}')
        else:
            print(f'\nAgent returned unexpected action type: {type(action).__name__}')
            print(f'Action details: {action}')

    except Exception as e:
        logger.error(f'Error during agent step: {e}', exc_info=True)


if __name__ == '__main__':
    main()
