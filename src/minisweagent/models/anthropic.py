from typing import Literal

from minisweagent.models.litellm_model import LitellmModel, LitellmModelConfig
from minisweagent.models.utils.cache_control import set_cache_control


class AnthropicModelConfig(LitellmModelConfig):
    set_cache_control: Literal["default_end"] | None = "default_end"
    """Set explicit cache control markers, for example for Anthropic models"""


class AnthropicModel(LitellmModel):
    """Anthropic model wrapper with cache control.

    This class provides a thin wrapper around LitellmModel with
    default cache control settings for Anthropic models.
    """

    def __init__(self, *, config_class: type = AnthropicModelConfig, **kwargs):
        super().__init__(config_class=config_class, **kwargs)

    def query(self, messages: list[dict], **kwargs) -> dict:
        messages = set_cache_control(messages, mode="default_end")
        return super().query(messages, **kwargs)
