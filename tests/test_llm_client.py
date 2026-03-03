from __future__ import annotations

from types import SimpleNamespace

from llm.mock_client import MockLLMClient
from llm.openai_client import OpenAILLMClient


class _FakeCompletions:
    def create(self, **kwargs):
        content = "ok"
        if kwargs.get("messages"):
            content = kwargs["messages"][-1].get("content", "ok")
        return SimpleNamespace(
            id="r1",
            model=str(kwargs.get("model") or "fake-model"),
            usage=SimpleNamespace(prompt_tokens=12, completion_tokens=6, total_tokens=18),
            choices=[SimpleNamespace(message=SimpleNamespace(content=content), finish_reason="stop")],
        )


class _FakeChat:
    def __init__(self):
        self.completions = _FakeCompletions()


class _FakeOpenAI:
    def __init__(self):
        self.chat = _FakeChat()


def test_openai_client_smoke() -> None:
    client = OpenAILLMClient(client=_FakeOpenAI())
    resp = client.chat_completion(
        model="gpt-5-mini",
        messages=[{"role": "user", "content": "hello"}],
        max_tokens=10,
        temperature=0.7,
    )
    assert resp.choices[0].message.content == "hello"
    assert resp.usage.total_tokens == 18


def test_mock_client_smoke() -> None:
    client = MockLLMClient()
    resp = client.chat_completion(model="mock", messages=[{"role": "user", "content": "Напиши код"}])
    assert resp.choices[0].message.content
    assert resp.usage.total_tokens > 0


if __name__ == "__main__":
    test_openai_client_smoke()
    test_mock_client_smoke()
    print("\n🎉 LLM client tests passed")
