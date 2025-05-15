import copy
import os
from collections import deque
from functools import partial

from litellm import ChatCompletionToolParam
from omegaconf import DictConfig

import openhands.agenthub.codeact_agent.function_calling as codeact_function_calling

# from openhands.agenthub.error_bench_agent.tools.llm_based_edit import LLMBasedFileEditTool
# from openhands.agenthub.error_bench_agent.tools.str_replace_editor import (
#     create_str_replace_editor_tool,
# )
from openhands.agenthub.codeact_agent.tools.think import ThinkTool
from openhands.agenthub.error_bench_agent.tools.bash import create_cmd_run_tool

# from openhands.agenthub.error_bench_agent.tools.browser import BrowserTool
from openhands.agenthub.error_bench_agent.tools.finish import FinishTool
from openhands.agenthub.error_bench_agent.tools.ipython import IPythonTool
from openhands.controller.agent import Agent
from openhands.controller.state.state import State
from openhands.core.config import AgentConfig
from openhands.core.logger import openhands_logger as logger
from openhands.core.message import Message
from openhands.events.action import (
    Action,
    AgentFinishAction,
)
from openhands.events.event import Event
from openhands.llm.llm import LLM
from openhands.memory.condenser import Condenser
from openhands.memory.condenser.condenser import Condensation, View
from openhands.memory.conversation_memory import ConversationMemory
from openhands.runtime.plugins import (
    AgentSkillsRequirement,
    JupyterRequirement,
    PluginRequirement,
)
from openhands.utils.prompt import PromptManager
import pandas as pd
from pathlib import Path

class ErrorAgent(Agent):
    VERSION = '2.2'
    """
    The ErrorAgent is based on the CodeActAgent.
    The agent works by passing the model a list of action-observation pairs and prompting the model to take the next step.

    ### Overview

    This agent implements the CodeAct idea ([paper](https://arxiv.org/abs/2402.01030), [tweet](https://twitter.com/xingyaow_/status/1754556835703751087)) that consolidates LLM agents' **act**ions into a unified **code** action space for both *simplicity* and *performance* (see paper for more details).

    The conceptual idea is illustrated below. At each turn, the agent can:

    1. **Converse**: Communicate with humans in natural language to ask for clarification, confirmation, etc.
    2. **CodeAct**: Choose to perform the task by executing code
    - Execute any valid Linux `bash` command
    - Execute any valid `Python` code with [an interactive Python interpreter](https://ipython.org/). This is simulated through `bash` command, see plugin system below for more details.

    ![image](https://github.com/All-Hands-AI/OpenHands/assets/38853559/92b622e3-72ad-4a61-8f41-8c040b6d5fb3)

    """

    sandbox_plugins: list[PluginRequirement] = [
        # NOTE: AgentSkillsRequirement need to go before JupyterRequirement, since
        # AgentSkillsRequirement provides a lot of Python functions,
        # and it needs to be initialized before Jupyter for Jupyter to use those functions.
        AgentSkillsRequirement(),
        JupyterRequirement(),
    ]

    def __init__(
        self,
        llm: LLM,
        config: AgentConfig,
        cfg: DictConfig | None = None,
    ) -> None:
        """Initializes a new instance of the CodeActAgent class.

        Parameters:
        - llm (LLM): The llm to be used by this agent
        - config (AgentConfig): The configuration for this agent
        """
        super().__init__(llm, config)
        self.cfg = cfg
        self.pending_actions: deque[Action] = deque()
        self.reset()
        self.tools = self._get_tools()
        self.prompt_manager = PromptManager(
            prompt_dir=os.path.join(os.path.dirname(__file__), 'prompts'),
        )
        # Create a ConversationMemory instance
        self.conversation_memory = ConversationMemory(self.config, self.prompt_manager)

        self.condenser = Condenser.from_config(self.config.condenser)
        logger.debug(f'Using condenser: {type(self.condenser)}')

        self.response_to_actions_fn = partial(
            codeact_function_calling.response_to_actions, cfg=self.cfg
        )

    def _get_tools(self) -> list[ChatCompletionToolParam]:
        # For these models, we use short tool descriptions ( < 1024 tokens)
        # to avoid hitting the OpenAI token limit for tool descriptions.
        SHORT_TOOL_DESCRIPTION_LLM_SUBSTRS = ['gpt-', 'o3', 'o1', 'o4']

        use_short_tool_desc = False
        if self.llm is not None:
            use_short_tool_desc = any(
                model_substr in self.llm.config.model
                for model_substr in SHORT_TOOL_DESCRIPTION_LLM_SUBSTRS
            )

        tools = []
        if self.config.enable_cmd:
            tools.append(create_cmd_run_tool(use_short_description=use_short_tool_desc))
        if self.config.enable_think:
            tools.append(ThinkTool)
        if self.config.enable_finish and not self.cfg.keep_going_until_succeed:
            tools.append(FinishTool)
        # if self.config.enable_browsing:
        #     tools.append(WebReadTool)
        #     tools.append(BrowserTool)
        if self.config.enable_jupyter:
            tools.append(IPythonTool)
        # if self.config.enable_llm_editor:
        #     tools.append(LLMBasedFileEditTool)
        # if self.config.enable_editor:
        #     tools.append(
        #         create_str_replace_editor_tool(
        #             use_short_description=use_short_tool_desc
        #         )
        #     )
        return tools

    def reset(self) -> None:
        """Resets the CodeAct Agent."""
        super().reset()
        self.pending_actions.clear()

    def step(self, state: State) -> Action:
        """Performs one step using the CodeAct Agent.

        This includes gathering info on previous steps and prompting the model to make a command to execute.

        Parameters:
        - state (State): used to get updated info

        Returns:
        - CmdRunAction(command) - bash command to run
        - IPythonRunCellAction(code) - IPython code to run
        - AgentDelegateAction(agent, inputs) - delegate action for (sub)task
        - MessageAction(content) - Message action to run (e.g. ask for clarification)
        - AgentFinishAction() - end the interaction
        """
        # Continue with pending actions if any
        if self.pending_actions:
            return self.pending_actions.popleft()

        # if we're done, go back
        latest_user_message = state.get_last_user_message()
        if latest_user_message and latest_user_message.content.strip() == '/exit':
            return AgentFinishAction()

        # Condense the events from the state. If we get a view we'll pass those
        # to the conversation manager for processing, but if we get a condensation
        # event we'll just return that instead of an action. The controller will
        # immediately ask the agent to step again with the new view.
        condensed_history: list[Event] = []
        match self.condenser.condensed_history(state):
            case View(events=events):
                condensed_history = events

            case Condensation(action=condensation_action):
                return condensation_action

        logger.debug(
            f'Processing {len(condensed_history)} events from a total of {len(state.history)} events'
        )

        if self.cfg.is_plotting_enabled:
            content = [x.content for x in condensed_history if 'content' in x.__dict__]
            # Find all the times where data:image/png;base64, appears in the text
            text = '\n'.join(content)
            pngs = []
            for i, line in enumerate(text.split('\n')):
                if 'data:image/png;base64,' in line:
                    # breakpoint()
                    # with open(current / f'{i}.png', 'wb') as f:
                    #     f.write(png.encode('utf-8'))
                    png = line.split('data:image/png;base64,')[1].split(')')[0]
                    pngs.append(png)
        messages = self._get_messages(condensed_history)
        params: dict = {
            'messages': self.llm.format_messages_for_llm(messages),
        }
        params['tools'] = self.tools

        if self.mcp_tools:
            # Only add tools with unique names
            existing_names = {tool['function']['name'] for tool in params['tools']}
            unique_mcp_tools = [
                tool
                for tool in self.mcp_tools
                if tool['function']['name'] not in existing_names
            ]

            if self.llm.config.model == 'gemini-2.5-pro-preview-03-25':
                logger.info(
                    f'Removing the default fields from the MCP tools for {self.llm.config.model} '
                    "since it doesn't support them and the request would crash."
                )
                # prevent mutation of input tools
                unique_mcp_tools = copy.deepcopy(unique_mcp_tools)
                # Strip off default fields that cause errors with gemini-preview
                for tool in unique_mcp_tools:
                    if 'function' in tool and 'parameters' in tool['function']:
                        if 'properties' in tool['function']['parameters']:
                            for prop_name, prop in tool['function']['parameters'][
                                'properties'
                            ].items():
                                if 'default' in prop:
                                    del prop['default']

            params['tools'] += unique_mcp_tools
        # log to litellm proxy if possible
        params['extra_body'] = {'metadata': state.to_llm_metadata(agent_name=self.name)}

        # Remove anything that is more than contains more than 100 digits in a row
        if self.cfg.disable_numbers:
            # Get the data if not already there
            if 'numberical_values' not in dir(self):
                # Get the data file

                # Open the file
                df = pd.read_csv(
                    f'evaluation/benchmarks/error_bench/tasks/{self.cfg.class_type}/{self.cfg.instance}/train.csv'
                )
                # Extract all the the numbers of the first five rows and convert them to a string

                self.numberical_values = df.values.flatten()[:1000].tolist()
            
            for message in params['messages']:
                #try:
                candidate = message.get('content', '')
                if isinstance(candidate, dict):
                    text = candidate.get('text', '')
                elif isinstance(candidate, list):
                    if len(candidate) == 0:
                        continue
                    else:
                        candidate = candidate[0]
                    if isinstance(candidate, dict):
                        text = candidate.get('text', '')
                    elif isinstance(candidate, str):
                        text = candidate
                    else:
                        breakpoint()
                # if 'gemini' in self.cfg.llm_config or 'llama' in self.cfg.llm_config:
                #     text = message.get('content', '')[0]['text']
                # elif 'claude' in self.cfg.llm_config:
                #     text = message['content']
                # else:
                #     raise ValueError('Unsupported LLM config')
                # text = message.get('content', '')[0]["text"]
                counter = 0
                # Check if the text contains 4 or more consecutive digits
                for num in self.numberical_values:
                    if str(num)[:5] in text:
                        counter += 1

                    if counter > 20:
                        if 'gemini' in self.cfg.llm_config:
                            # Replace the message with a placeholder
                            message['content'][0]['text'] = (
                                'Raw numbers of the dataset not available. Report this error to the user and keep going.'
                            )
                        elif 'claude' in self.cfg.llm_config:
                            # Replace the message with a placeholder
                            message['content'] = (
                                'Raw numbers of the dataset not available. Report this error to the and keep going.'
                            )
                        # Replace the message with a placeholder
                        # message['content'][0]['text'] = "Raw numbers of the dataset not available. Report this to the user and keep going."
                        break
                # If the text contains 4 or more consecutive digits, replace it with "Result not available"
                # if counter > 4:

                # # If the text contains 100 or more consecutive digits, replace it with "Result not available"
                # if long_digits_pattern.search(text):
                #     # Replace the message with a placeholder
                #     message['content'][0]['text'] = "Raw numbers of the dataset not available"

                # # if "0.344435  1.548645" in text:
                # #     # Replace the message with a placeholder
                # #     message['content'][0]['text'] = "Raw numbers of the dataset not available"

                # filtered.append(message)
                # # If there are more than 100 digits in a row, remove the message
                # #if (

                # # message['content'][0]['text']
        # Go over the messages and if there is any with more  tokens, don't visualize it

        # Load all the images
        # self.uuid4

        # Find each message with "already displayed to user" and remove it
        if self.cfg.is_plotting_enabled:
            
            png_iter = iter(pngs)
            def save_png(png_iter):
                # Save all the images in a list inside the evaluation folder
                images = Path(self.cfg.eval_output_dir) / 'images'
                images.mkdir(parents=True, exist_ok=True)
                # Save the images in the folder
                for i, b64 in enumerate(pngs):
                    import base64
                    import pathlib

                    # Convert the base64 string to bytes
                    img_bytes = base64.b64decode(b64)
                    pathlib.Path(images / f'{i}.png').write_bytes(img_bytes)
            save_png(png_iter)
            
            for idx, message in enumerate(params['messages']):
                rebuilt = []
                # rebuild this message’s content
                if isinstance(message['content'], list) and (
                    'gemini' in self.cfg.llm_config or 'gpt' in self.cfg.llm_config or 'llama' in self.cfg.llm_config
                ):
                    for part in message['content']:
                        if (
                            part.get('type') == 'text'
                            and 'already displayed to user' in part['text']
                        ):
                            stripped = (
                                part['text']
                                .replace('already displayed to user', '')
                                .strip()
                            )
                            if stripped:
                                rebuilt.append({'type': 'text', 'text': stripped})
                            try:
                                img_b64 = next(png_iter)
                            except StopIteration as e:
                                raise ValueError(
                                    'Not enough images in `pngs` for every placeholder.'
                                ) from e

                            rebuilt.append(
                                {
                                    'type': 'image_url',
                                    'image_url': {
                                        'url': f'data:image/png;base64,{img_b64}',
                                        'format': 'image/png',
                                    },
                                }
                            )
                        else:
                            rebuilt.append(part)  # unchanged part
                    message['content'] = rebuilt

                elif isinstance(message, dict) and 'claude' in self.cfg.llm_config or "gpt" in self.cfg.llm_config or "llama" in self.cfg.llm_config:
                    if (
                        'content' in message
                        and 'already displayed to user' in message['content']
                    ):
                        # Check how many times "already displayed to user" appears
                        cnt = message['content'].count('already displayed to user')
                        # 1️⃣ strip the marker
                        stripped = (
                            message['content']
                            .replace('already displayed to user', '')
                            .strip()
                        )
                        if stripped:
                            rebuilt.append({'type': 'text', 'text': stripped})

                        for _ in range(cnt):
                            # 2️⃣ inject next PNG
                            try:
                                img_b64 = next(png_iter)
                            except StopIteration as e:
                                raise ValueError(
                                    'Not enough images in `pngs` for every placeholder.'
                                ) from e
                            # rebuilt.append({"type": "text", "text": f"Image {i}"})
                            rebuilt.append(
                                {
                                    'type': 'image_url',
                                    'image_url': {
                                        'url': f'data:image/png;base64,{img_b64}',
                                        'format': 'image/png',
                                    },
                                }
                            )
                
                        params['messages'][idx] = {
                            'content': rebuilt,
                            'role': message['role'],
                        }  # put rebuilt list back
                        # ← put rebuilt list back
                else:
                    raise ValueError('Unsupported LLM config')
            # clone *after* the edits so the two dicts differ only by this change
            new_msx = copy.deepcopy(params)
        else:
            new_msx = params

        response = self.llm.completion(**new_msx)

        logger.debug(f'Response from LLM: {response}')
        actions = self.response_to_actions_fn(response)
        logger.debug(f'Actions after response_to_actions: {actions}')
        for action in actions:
            self.pending_actions.append(action)
        return self.pending_actions.popleft()

    def _get_messages(self, events: list[Event]) -> list[Message]:
        """Constructs the message history for the LLM conversation.

        This method builds a structured conversation history by processing events from the state
        and formatting them into messages that the LLM can understand. It handles both regular
        message flow and function-calling scenarios.

        The method performs the following steps:
        1. Checks for SystemMessageAction in events, adds one if missing (legacy support)
        2. Processes events (Actions and Observations) into messages, including SystemMessageAction
        3. Handles tool calls and their responses in function-calling mode
        4. Manages message role alternation (user/assistant/tool)
        5. Applies caching for specific LLM providers (e.g., Anthropic)
        6. Adds environment reminders for non-function-calling mode

        Args:
            events: The list of events to convert to messages

        Returns:
            list[Message]: A list of formatted messages ready for LLM consumption, including:
                - System message with prompt (from SystemMessageAction)
                - Action messages (from both user and assistant)
                - Observation messages (including tool responses)
                - Environment reminders (in non-function-calling mode)

        Note:
            - In function-calling mode, tool calls and their responses are carefully tracked
              to maintain proper conversation flow
            - Messages from the same role are combined to prevent consecutive same-role messages
            - For Anthropic models, specific messages are cached according to their documentation
        """
        if not self.prompt_manager:
            raise Exception('Prompt Manager not instantiated.')

        # Use ConversationMemory to process events (including SystemMessageAction)
        messages = self.conversation_memory.process_events(
            condensed_history=events,
            max_message_chars=self.llm.config.max_message_chars,
            vision_is_active=self.llm.vision_is_active(),
        )

        if self.llm.is_caching_prompt_active():
            self.conversation_memory.apply_prompt_caching(messages)

        return messages
