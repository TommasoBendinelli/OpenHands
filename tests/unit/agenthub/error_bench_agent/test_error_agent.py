# /workspace/OpenHands/tests/unit/agenthub/error_bench_agent/test_error_agent.py
import copy
import base64
from unittest.mock import MagicMock, patch

import pytest
from omegaconf import DictConfig

from openhands.agenthub.error_bench_agent.error_agent import ErrorAgent
from openhands.controller.state.state import State
from openhands.core.config import AgentConfig, LLMConfig
from openhands.events.event import EventSource
from openhands.core.message import Message  # For type hinting if needed
from openhands.events.action import MessageAction, Action
from openhands.events.observation import Observation # Generic observation
from openhands.llm.llm import LLM

# Minimal LLMConfig for tests
MINIMAL_LLM_CONFIG = LLMConfig(model='test_model_error_agent')

# Minimal AgentConfig for tests - instantiating with defaults
# AgentConfig defines capabilities, not instance-specific attributes like name or llm object.
MINIMAL_AGENT_CONFIG = AgentConfig()

DUMMY_PNG_B64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="

@pytest.fixture
def error_agent_fixture(llm_config_override: LLMConfig | None = None, agent_config_override: AgentConfig | None = None):
    current_llm_config = copy.deepcopy(llm_config_override or MINIMAL_LLM_CONFIG)
    current_agent_config = copy.deepcopy(agent_config_override or MINIMAL_AGENT_CONFIG)



    mock_llm = MagicMock(spec=LLM)
    # LLMConfig instance that the agent will use for its config.vision_is_active() call
    # We will mock vision_is_active on this specific instance in tests that need it.
    agent_llm_config_instance = LLMConfig(model=current_llm_config.model, disable_vision=current_llm_config.disable_vision) 
    mock_llm.config = agent_llm_config_instance
    
    error_bench_specific_cfg = DictConfig({'keep_going_until_succeed': False, 'is_plotting_enabled': False, 'disable_numbers': False})
    agent = ErrorAgent(llm=mock_llm, config=current_agent_config, cfg=error_bench_specific_cfg)
    agent.response_to_actions_fn = MagicMock(return_value=[MessageAction(content="mocked_error_action")])
    
    return agent

@pytest.fixture
def error_initial_state_fixture():
    state = State(session_id="test_error_session")
    msg_action = MessageAction(content="Initial user prompt with <image> placeholder.")
    msg_action._source = "user"
    state.history = [msg_action]
    return state

@pytest.mark.parametrize(
    "llm_model_name, set_vision_active_on_config, initial_message_content, expect_value_error, expected_llm_content_if_no_error",
    [
        ("gpt-4", False, "Text without placeholder.", False, "Text without placeholder."),
        ("gpt-4-vision-preview", True, "Text with <image> placeholder.", True, None), # Expect ValueError
        ("gpt-4-vision-preview", True, "No <image> placeholder here.", False, "No <image> placeholder here."),
        ("gpt-4", False, "Text with <image> placeholder.", False, "Text with <image> placeholder."),
    ]
)
def test_error_agent_step_llm_call_params(
    llm_model_name, set_vision_active_on_config, initial_message_content, expect_value_error, expected_llm_content_if_no_error,
    error_initial_state_fixture: State # This fixture provides a State with a default history. We will override history.
):
    current_agent_config = copy.deepcopy(MINIMAL_AGENT_CONFIG)

    mock_llm_for_agent = MagicMock(spec=LLM)
    actual_llm_config_instance = LLMConfig(model=llm_model_name, disable_vision=not set_vision_active_on_config)
    mock_llm_for_agent.config = actual_llm_config_instance
    mock_llm_for_agent.vision_is_active = MagicMock(return_value=set_vision_active_on_config)

    error_bench_specific_cfg_for_test = DictConfig({'keep_going_until_succeed': False, 'is_plotting_enabled': False, 'disable_numbers': False})
    agent = ErrorAgent(llm=mock_llm_for_agent, config=current_agent_config, cfg=error_bench_specific_cfg_for_test)
    agent.response_to_actions_fn = MagicMock(return_value=[MessageAction(content="mocked_error_action_param")])

    # Use a fresh state for each parameterized run, setting history directly
    state = State(session_id=f"test_session_{llm_model_name}_{initial_message_content[:10]}") 
    msg_action = MessageAction(content=initial_message_content)
    msg_action._source = "user" 
    state.history = [msg_action]

    if expect_value_error:
        with pytest.raises(ValueError, match="Not enough images provided for <image> placeholder in vision model."):
            agent.step(state)
        # If an error is expected due to <image> tag, vision_is_active should have been called.
        if "<image>" in initial_message_content: # This check is redundant if expect_value_error is true due to image
            mock_llm_for_agent.vision_is_active.assert_called_once()
        return

    # If no ValueError is expected
    action_result = agent.step(state)

    mock_llm_for_agent.completion.assert_called_once()
    _called_args, called_kwargs = mock_llm_for_agent.completion.call_args
    
    called_messages = called_kwargs['messages']
    
    user_message_found_and_correct = False
    for msg_dict in called_messages: # Iterate over the list of message dictionaries
        if msg_dict.get('role') == 'user': # Use .get for safety, though role should exist
            # Ensure content is a string before comparison, as vision models might make it a list
            called_content = msg_dict.get('content', '')
            assert called_content == expected_llm_content_if_no_error, \
                f"User message content mismatch. Got: {called_content}, Expected: {expected_llm_content_if_no_error}"
            user_message_found_and_correct = True
            break
    assert user_message_found_and_correct, f"User message not found or content incorrect in LLM call. Called messages: {called_messages}"

    # Check metadata
    assert 'metadata' in called_kwargs.get('extra_body', {}), "metadata not in extra_body"
    assert called_kwargs['extra_body']['metadata'].get('agent_name') == agent.name, "agent_name in metadata mismatch"

    assert action_result == MessageAction(content="mocked_error_action_param")
    agent.response_to_actions_fn.assert_called_once_with(mock_llm_for_agent.completion.return_value)
    
    # Check if vision_is_active was called appropriately by _get_messages
    if "<image>" in initial_message_content: 
        mock_llm_for_agent.vision_is_active.assert_called_once()
    else: 
        mock_llm_for_agent.vision_is_active.assert_not_called()


def test_error_agent_step_non_vision_path(error_agent_fixture: ErrorAgent, error_initial_state_fixture: State):
    agent = error_agent_fixture
    state = error_initial_state_fixture
    
    # Ensure vision_is_active on the agent's llm.config returns False
    agent.llm.vision_is_active = MagicMock(return_value=False)

    msg_action = MessageAction(content="Simple text, no vision.")
    msg_action._source = "user"
    state.history = [msg_action]
    expected_messages = agent._get_messages(state.history)
    expected_params = {'messages': expected_messages}

    agent.step(state)

    # Placeholder for new assertions
    agent.response_to_actions_fn.assert_called_once_with(agent.llm.completion.return_value)
    agent.llm.vision_is_active.assert_not_called() # Should not be called if no <image> tag, or if called, it's fine if it returns False
