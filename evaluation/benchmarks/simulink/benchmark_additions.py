import fcntl
import json
import os
from pathlib import Path
from typing import Callable

from openhands.controller.state.state import State
from openhands.events.action import Action, MessageAction


def safe_append(path: Path, text: str):
    # Open in append+ mode so writes always go to end
    with path.open('a+') as f:
        # Acquire exclusive lock (blocks until free)
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            f.write(text)
            f.flush()  # ensure it hits disk
        finally:
            # Release the lock
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def kill_instance(output_file: str):
    """Terminate the runtime container recorded in the output file.
    If the output does not contain a ``sid`` field, all containers with the
    ``openhands-runtime-`` prefix will be killed instead.
    """

    sid_to_kill = None
    with open(output_file, 'r') as f:
        for line in f:
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            if 'sid' in data and data['sid']:
                sid_to_kill = f'openhands-runtime-{data["sid"]}'
                break

    if sid_to_kill:
        print('sid to kill: ', sid_to_kill)
        os.system(f"docker ps -q --filter 'name={sid_to_kill}' | xargs -r docker kill")
    else:
        raise ValueError('SID not found')


def errorbench_user_response(
    state: State,
    encapsulate_solution: bool = False,
    try_parse: Callable[[Action], str] | None = None,
) -> str:
    msg = 'Please continue working on the task and submit a new solution to the user via compute_metrics.py. Your result is not good enough yet.\n'
    if state.history:
        from openhands.events.observation import CmdOutputObservation

        msg_content = [
            x.content
            for x in state.history
            if (
                isinstance(x, CmdOutputObservation)
                and 'Congratulations! You have reached the accuracy threshold of 1.0.'
                in x.content
            )
        ]
        if len(msg_content) > 0:
            return '/exit'

        # check if the last action has an answer, if so, early exit
        if try_parse is not None:
            last_action = next(
                (
                    event
                    for event in reversed(state.history)
                    if isinstance(event, Action)
                ),
                None,
            )
            ans = try_parse(last_action)
            if ans is not None:
                return '/exit'

        # check if the agent has tried to talk to the user 3 times, if so, let the agent know it can give up
        [
            event
            for event in state.history
            if isinstance(event, MessageAction) and event.source == 'user'
        ]
        # if len(user_msgs) >= 2:
        #     # let the agent know that it can give up when it has tried 3 times
        #     return (
        #         msg
        #         + 'If you want to give up, use the "finish" tool to finish the interaction.\n'
        #     )
    return msg
