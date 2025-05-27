import fcntl
import json
import os
from pathlib import Path


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
    with open(output_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            assert 'sid' in data, f"Missing 'sid' in line: {line}"
            sid_to_kill = 'openhands-runtime-' + data['sid']
            break

    print('sid to kill: ', sid_to_kill)
    print(f"docker ps -q --filter 'name={sid_to_kill}'")
    os.system(f"docker ps -q --filter 'name={sid_to_kill}' | xargs -r docker kill")


def errorbench_user_response(
    state: State,
    encapsulate_solution: bool = False,
    try_parse: Callable[[Action], str] | None = None,
) -> str:
    encaps_str = (
        (
            'Please encapsulate your final answer (answer ONLY) within <solution> and </solution>.\n'
            'For example: The answer to the question is <solution> The index 42 value is clearly an outlier </solution>.\n'
        )
        if encapsulate_solution
        else ''
    )
    msg = (
        'Please continue working on the task and submit a new solution to the user via compute_metrics.py. Your result is not good enough yet.\n'
    )
    if state.history:
        from openhands.events.observation import CmdOutputObservation
        import re
        msg_content = [x.content for x in state.history if (isinstance(x, CmdOutputObservation) and "Congratulations! You have reached the accuracy threshold of 1.0." in x.content )]
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
        user_msgs = [
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

