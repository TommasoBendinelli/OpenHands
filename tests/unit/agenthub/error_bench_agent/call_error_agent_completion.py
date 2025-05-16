# /workspace/OpenHands/call_error_agent_completion.py


import argparse
import os
import sys

import toml
from omegaconf import OmegaConf
from dataclasses import fields
from unittest.mock import MagicMock

from openhands.agenthub.error_bench_agent.error_agent import ErrorAgent
from openhands.core.config import AgentConfig, LLMConfig
from openhands.core.logger import openhands_logger as logger
from openhands.llm.llm import LLM
from openhands.controller.state.state import State
from openhands.core.message import Message, TextContent
from openhands.events.action import MessageAction
from openhands.events.event import EventSource

# Open hydra



def main():
    parser = argparse.ArgumentParser(description="Run ErrorAgent with specified LLM configurations.")
    parser.add_argument(
        "llm_configs",
        type=str,
        help="Comma-separated list of LLM configuration names to run (e.g., 'gpt-4o,gemini_pro')."
    )
    parser.add_argument(
        "--config-file",
        type=str,
        default="/workspace/OpenHands/config.toml",
        help="Path to the TOML configuration file for LLMs."
    )
    parser.add_argument(
        "--temperature-override",
        type=float,
        default=None,
        help="Temperature override for the LLM."
    )
    parser.add_argument(
        "--hydra-config-file",
        type=str,
        default=None,
        help="Path to a Hydra YAML configuration file for the agent's 'cfg' object."
    )
    parsed_args = parser.parse_args()
    logger.info(f"Starting script with parsed arguments: {parsed_args}")

    raw_input_str = parsed_args.llm_configs
    actual_configs_str = raw_input_str

    # Check if the input string is in key="value" or key=value format
    if '=' in raw_input_str:
        parts = raw_input_str.split('=', 1)
        if len(parts) == 2:
            # parts[0] is the key (e.g., "llm-configs")
            # parts[1] is the value string (e.g., '"gpt-4o"' or 'gpt-4o,gpt-3.5-turbo')
            value_str = parts[1]
            # Remove surrounding quotes from the value string, if any
            if (value_str.startswith('"') and value_str.endswith('"')) or \
               (value_str.startswith("'") and value_str.endswith("'")):
                actual_configs_str = value_str[1:-1]
            else:
                actual_configs_str = value_str
        # If there's an '=' but not in key=value format, we might still want to log a warning
        # or treat it as a literal if that's a possible LLM name. For now, this handles simple key=value.
    
    # Now, actual_configs_str should contain the comma-separated list of LLM names
    llm_names_to_run = [name.strip() for name in actual_configs_str.split(',') if name.strip()]

    if not llm_names_to_run:
        logger.error(f"No valid LLM configuration names could be extracted from input: '{raw_input_str}'. Please provide a comma-separated list of LLM names, e.g., 'gpt-4o' or 'llm-configs=\"model_name\".'")
        return
    
    logger.info(f"Successfully parsed LLM configuration names to run: {llm_names_to_run} (from input: '{raw_input_str}')")

    GEMINI_PRO_DEFAULTS = {
        'model': 'gemini-1.5-flash-latest',
        'temperature': 0.1,
        'custom_llm_provider': 'gemini',
    }

    for current_llm_name in llm_names_to_run:
        logger.info(f"\n===== Running for LLM: {current_llm_name} =====")
        try:
            # Initialize cfg: Load from hydra_config_file or use fallback
            cfg = None
            if parsed_args.hydra_config_file:
                if os.path.exists(parsed_args.hydra_config_file):
                    logger.info(f'Loading Hydra configuration from: {parsed_args.hydra_config_file}')
                    try:
                        cfg = OmegaConf.load(parsed_args.hydra_config_file)
                    except Exception as e:
                        logger.error(f'Failed to load Hydra config file: {e}', exc_info=True)
                        logger.warning("Proceeding with default cfg due to Hydra load failure.")
                        # cfg remains None, will use fallback
                else:
                    logger.warning(f"Hydra configuration file '{parsed_args.hydra_config_file}' not found.")
            
            if cfg is None:
                logger.warning("Hydra cfg is None or not loaded. Initializing with default ErrorAgent cfg.")
                cfg = OmegaConf.create({
                    # 'llm_config' will be set by current_llm_name shortly
                    'keep_going_until_succeed': False,
                    'max_retries': 3,
                    'is_plotting_enabled': False,
                    'disable_numbers': False,
                })
            
            # Set the llm_config for the current iteration
            cfg.llm_config = current_llm_name
            logger.info(f"Using cfg for {current_llm_name}: {OmegaConf.to_container(cfg)}")

            # LLM Configuration and Instantiation
            selected_llm_config_dict = None
            full_toml_data = None
            
            config_file_exists = os.path.exists(parsed_args.config_file)
            if config_file_exists:
                logger.info(f'Loading LLM configuration from: {parsed_args.config_file}')
                try:
                    full_toml_data = toml.load(parsed_args.config_file)
                except Exception as e:
                    logger.error(f"Error loading TOML file {parsed_args.config_file}: {e}", exc_info=True)
                    # full_toml_data remains None
            else:
                logger.warning(f"LLM Configuration file '{parsed_args.config_file}' not found.")

            # cfg.llm_config (which is current_llm_name) is the primary selector
            target_llm_name_from_cfg = cfg.llm_config 

            if target_llm_name_from_cfg and full_toml_data:
                logger.info(f"Attempting to load LLM configuration for: '{target_llm_name_from_cfg}' from TOML")
                named_llm_configs = full_toml_data.get('llm', {})
                if target_llm_name_from_cfg in named_llm_configs and isinstance(named_llm_configs[target_llm_name_from_cfg], dict):
                    selected_llm_config_dict = named_llm_configs[target_llm_name_from_cfg].copy()
                    logger.info(f"Successfully loaded LLM configuration '{target_llm_name_from_cfg}' from [llm.{target_llm_name_from_cfg}] section.")
                else:
                    logger.warning(f"LLM configuration '{target_llm_name_from_cfg}' not found as a dictionary under [llm] in '{parsed_args.config_file}'. Will check default section or use fallbacks.")

            if selected_llm_config_dict is None and full_toml_data:
                logger.info(f"No specific [llm.{target_llm_name_from_cfg}] section found or it was invalid. Attempting to load direct keys from general [llm] section.")
                default_llm_section_direct_keys = {
                    k: v for k, v in full_toml_data.get('llm', {}).items() if not isinstance(v, dict)
                }
                if default_llm_section_direct_keys.get('model'):
                    selected_llm_config_dict = default_llm_section_direct_keys.copy()
                    # If current_llm_name is a model string, it might be intended to override the model from the default section
                    # For now, we assume the default section is a generic fallback. If specific model override is needed here, logic can be added.
                    logger.info("Successfully loaded model configuration from direct keys of [llm] section as a fallback.")
                else:
                    logger.warning("General [llm] section in config file does not directly specify a 'model' or is missing.")
            
            if selected_llm_config_dict is None:
                logger.warning(f"No suitable configuration found in TOML for '{current_llm_name}'. Using Gemini Pro defaults as fallback, and '{current_llm_name}' as model name if it differs.")
                selected_llm_config_dict = GEMINI_PRO_DEFAULTS.copy()
                # Use current_llm_name as the model if it's not the gemini default model name, assuming it's a valid model identifier
                if current_llm_name != GEMINI_PRO_DEFAULTS['model']:
                    selected_llm_config_dict['model'] = current_llm_name
                # Apply environment variable overrides for Gemini defaults if still using Gemini model
                if selected_llm_config_dict['model'] == GEMINI_PRO_DEFAULTS['model']:
                     selected_llm_config_dict['model'] = os.getenv('OPENHANDS_LLM_MODEL', selected_llm_config_dict['model'])
                     selected_llm_config_dict['temperature'] = float(os.getenv('OPENHANDS_LLM_TEMPERATURE', selected_llm_config_dict['temperature']))

            if parsed_args.temperature_override is not None:
                selected_llm_config_dict['temperature'] = parsed_args.temperature_override
                logger.info(f'Overriding temperature to: {parsed_args.temperature_override}')

            if not selected_llm_config_dict or not selected_llm_config_dict.get('model'):
                logger.critical(f"LLM 'model' could not be determined for '{current_llm_name}'. Skipping this LLM.")
                continue 

            selected_llm_config_dict['temperature'] = float(selected_llm_config_dict.get('temperature', 0.1))
            
            valid_llm_config_fields = set(LLMConfig.model_fields.keys())
            llm_config_args = {
                k: v for k, v in selected_llm_config_dict.items() if k in valid_llm_config_fields
            }

            if 'model' not in llm_config_args:
                logger.critical(f"LLM 'model' is missing in the final config dict for '{current_llm_name}'. Skipping this LLM.")
                continue
                
            logger.info(f"Final dictionary for LLMConfig ({current_llm_name}): {llm_config_args}")

            llm_instance = None
            try:
                llm_config_obj = LLMConfig(**llm_config_args)
                logger.info(f"LLMConfig created for {current_llm_name}: model='{llm_config_obj.model}', temperature='{getattr(llm_config_obj, 'temperature', 'N/A')}', provider='{getattr(llm_config_obj, 'custom_llm_provider', 'default')}'")
                llm_instance = LLM(config=llm_config_obj)
                logger.info(f"Real LLM instance created for {current_llm_name}.")
            except Exception as e:
                logger.critical(f"Failed to create LLMConfig or LLM instance for {current_llm_name}: {e}", exc_info=True)
                continue

            agent_config = AgentConfig() 
            logger.info(f'AgentConfig initialized (default) for {current_llm_name}.')
            
            error_agent_instance = None
            try:
                error_agent_instance = ErrorAgent(llm=llm_instance, config=agent_config, cfg=cfg)
                logger.info(f'ErrorAgent instantiated successfully for {current_llm_name}.')
            except Exception as e:
                logger.error(f'Failed to instantiate ErrorAgent for {current_llm_name}: {e}', exc_info=True)
                continue 
            
            prompt = 'What is a large language model? Explain in one sentence.'
            logger.info(f"Simulating user message with prompt for {current_llm_name}: '{prompt}'")
            
            state = State()
            user_message_action = MessageAction(content=prompt)
            state.history.append(user_message_action)
            
            action_result = error_agent_instance.step(state)
            
            logger.info(f'Agent ({current_llm_name}) returned action: {type(action_result).__name__}')
            
            if isinstance(action_result, MessageAction):
                print(f'\nLLM ({current_llm_name}) Output:\n{action_result.content}')
            elif isinstance(action_result, Message): 
                 print(f'\nLLM ({current_llm_name}) Output (Message type):\n{action_result.content}')
            else:
                print(f'\nAgent ({current_llm_name}) returned unexpected action type: {type(action_result).__name__}')
                print(f'Action details: {action_result}')

        except Exception as e:
            logger.error(f"!!!!! UNHANDLED EXCEPTION for LLM: {current_llm_name} - {type(e).__name__}: {e} !!!!!", exc_info=True)
            # Continue to the next LLM in the outer loop

if __name__ == '__main__':
    main()
