from agentdojo.functions_runtime import TaskEnvironment
from agentdojo.types import ChatAssistantMessage, ChatMessage, text_content_block_from_string


class AbortAgentError(Exception):
    """Checking tools can raise this exception to abort the execution
    of the agent pipeline (e.g. when a prompt injection is detected).
    """

    messages: list[ChatMessage]
    task_environment: TaskEnvironment

    def __init__(
        self,
        message: str,
        messages: list[ChatMessage],
        task_environment: TaskEnvironment,
    ):
        super().__init__(message)
        self.messages = [
            *messages,
            ChatAssistantMessage(
                role="assistant",
                content=[text_content_block_from_string(message)],
                tool_calls=[],
            ),
        ]
        self.task_environment = task_environment
