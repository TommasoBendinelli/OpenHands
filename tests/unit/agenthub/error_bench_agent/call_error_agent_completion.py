# /workspace/OpenHands/call_error_agent_completion.py

import argparse
import os
import sys

import toml
from omegaconf import OmegaConf

try:
    from openhands.agenthub.error_bench_agent.error_agent import ErrorAgent
    from openhands.core.config import AgentConfig, LLMConfig
    from openhands.core.logger import openhands_logger as logger
    from openhands.llm.llm import LLM
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

    args = parser.parse_args()
    logger.info(
        f"Starting script. Config File: '{args.config_file}', LLM Name: '{args.llm_name if args.llm_name else '(default handling)'}', Temp Override: {args.temperature_override}"
    )

    selected_llm_config_dict = None
    full_toml_data = None

    # Open hydra config too
    cfg = OmegaConf.load(args.config_file)

    GEMINI_PRO_DEFAULTS = {
        'model': 'gemini-1.5-flash-latest',
        'temperature': 0.1,
        'custom_llm_provider': 'gemini',
    }

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
        logger.info(
            f"Note: 'custom_llm_provider' found ('{selected_llm_config_dict['custom_llm_provider']}'). This is typically handled by LiteLLM, not directly by LLMConfig."
        )

    llm_config = LLMConfig(**final_llm_args_for_config_class)
    logger.info(
        f"LLMConfig created: model='{llm_config.model}', temperature='{getattr(llm_config, 'temperature', 'N/A')}'"
    )

    breakpoint()
    try:
        llm = LLM(config=llm_config)
        logger.info(
            f"LLM initialized: model='{llm.config.model}', temperature='{getattr(llm.config, 'temperature', 'N/A')}'"
        )
    except Exception as e:
        logger.error(f'Failed to initialize LLM with config: {e}', exc_info=True)
        return

    # MODIFIED AgentConfig initialization
    agent_config = AgentConfig()
    logger.info('AgentConfig initialized (default).')

    try:
        error_agent_instance = ErrorAgent(llm=llm, config=agent_config, cfg=None)
        logger.info('ErrorAgent instantiated successfully.')
    except Exception as e:
        logger.error(f'Failed to instantiate ErrorAgent: {e}', exc_info=True)
        return

    prompt = 'What is a large language model? Explain in one sentence.'
    logger.info(f"Calling trigger_manual_llm_completion with prompt: '{prompt}'")

    try:
        response = error_agent_instance.trigger_manual_llm_completion(
            prompt_text=prompt
        )
    except Exception as e:
        logger.error(f'Error calling trigger_manual_llm_completion: {e}', exc_info=True)
        return

    logger.info('Response from LLM completion:')
    if response and isinstance(response, dict):
        if 'error' in response:
            logger.error(
                f"LLM completion method returned an error: {response['error']}"
            )
            if 'details' in response:
                logger.error(f"Details: {response['details']}")
        elif (
            'choices' in response
            and response['choices']
            and isinstance(response['choices'], list)
            and len(response['choices']) > 0
            and response['choices'][0]
            and isinstance(response['choices'][0], dict)
            and 'message' in response['choices'][0]
            and isinstance(response['choices'][0]['message'], dict)
            and 'content' in response['choices'][0]['message']
        ):
            content = response['choices'][0]['message']['content']
            print(f'\nLLM Output:\n{content}')
        else:
            print(
                f'\nLLM Response (structure not fully parsed/unexpected):\n{response}'
            )
    else:
        print(
            f'\nReceived unexpected response format from trigger_manual_llm_completion:\n{response}'
        )


if __name__ == '__main__':
    main()
