import os

import numpy as np
import pytest

from openhands.core.config import OpenHandsConfig, load_from_toml
from openhands.core.message import ImageContent, Message, TextContent
from openhands.llm.llm import LLM
from openhands.runtime.browser.base64 import image_to_png_base64_url

TEST_IN_CI = os.getenv('TEST_IN_CI', 'False').lower() in ['true', '1', 'yes']


@pytest.mark.skipif(
    not TEST_IN_CI,
    reason='This test requires network access and Docker to run.',
)
def test_llm_image_and_text() -> None:
    """Ensure the model can process image and text together."""
    cfg = OpenHandsConfig()
    load_from_toml(cfg, 'config.toml')

    if 'gpt-o4-mini' in cfg.llms:
        llm_cfg = cfg.get_llm_config('gpt-o4-mini')
    elif 'gpt4o-mini' in cfg.llms:
        llm_cfg = cfg.get_llm_config('gpt4o-mini')
    else:
        pytest.skip('No gpt-o4-mini config found in config.toml')

    api_key = (
        llm_cfg.api_key.get_secret_value() if llm_cfg.api_key is not None else None
    )
    if not api_key or api_key == 'here':
        pytest.skip('No valid API key for vision test')

    # Create a simple red square image
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    img[:, :] = [255, 0, 0]
    img_url = image_to_png_base64_url(img, add_data_prefix=True)

    llm = LLM(llm_cfg)
    prompt = (
        "Is the square in the image red or blue? Respond with 'red' or 'blue' only."
    )
    message = Message(
        role='user',
        content=[TextContent(text=prompt), ImageContent(image_urls=[img_url])],
    )
    formatted = llm.format_messages_for_llm([message])
    response = llm.completion(messages=formatted)
    answer = response['choices'][0]['message']['content'].lower()
    assert 'red' in answer
