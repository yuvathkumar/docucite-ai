"""Tests for prompts, answer generation and the `/ask` and `/questions` endpoints."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.llm import GenerationError, UnavailableChat
from app.llm.fake import FakeChatModel, FakeEmbedder
from app.rag.answer import NOT_FOUND_MESSAGE, Answer, answer_question
from app.rag.embeddings import embed_missing_chunks
from app.rag.ingest import ingest_pdf
from app.rag.prompts import extract_citation_numbers, remove_invalid_citations
from tests.pdf_factory import make_pdf
from tests.test_auth import NEW_PASSWORD, make_user

PAGES = [
    "Vacation policy: employees accrue two vacation days per month of service.",
    "Expense policy: submit expense reports before the fifth of each month.",
]


def load_documents(session: Session) -> None:
    """Ingest and embed the sample document."""
    document = ingest_pdf(
        session, "handbook.pdf", make_pdf(PAGES), max_bytes=10**6, max_chars=300, overlap=30
    )
    embed_missing_chunks(session, FakeEmbedder(), document.id)


def ask_with(
    session: Session, chat: FakeChatModel, question: str = "vacation days per month"
) -> Answer:
    """Call ``answer_question`` with the fake embedder."""
    return answer_question(session, FakeEmbedder(), chat, question, top_k=3)


def test_extract_citation_numbers_ignores_invalid_and_duplicates() -> None:
    """Keep only real passage numbers, once each, in order."""
    assert extract_citation_numbers("A [2]. B [1][2]. C [9]. D [0].", passage_count=3) == [2, 1]


def test_grouped_citations_are_understood() -> None:
    """Treat ``[1, 3]`` as citations of passages 1 and 3, and split it into one marker each."""
    assert extract_citation_numbers("Both facts [1, 3].", passage_count=3) == [1, 3]
    assert remove_invalid_citations("Both facts [1, 3].", passage_count=3) == "Both facts [1][3]."
    assert remove_invalid_citations("Mixed [2, 9].", passage_count=3) == "Mixed [2]."


def test_remove_invalid_citations_keeps_valid_ones() -> None:
    """Strip markers that point to passages that do not exist."""
    assert (
        remove_invalid_citations("Fact [1] and fake [9].", passage_count=2) == "Fact [1] and fake."
    )


def test_answer_cites_the_real_passage(db_session: Session) -> None:
    """Return the document, page and exact passage behind a citation."""
    load_documents(db_session)
    chat = FakeChatModel("Employees get two vacation days per month. [1]")

    answer = ask_with(db_session, chat)

    assert answer.found
    assert answer.citations[0].filename == "handbook.pdf"
    assert answer.citations[0].page_number == 1
    assert "two vacation days" in answer.citations[0].passage


def test_prompt_numbers_passages_and_warns_against_injection(db_session: Session) -> None:
    """Send numbered passages and tell the model to treat them as untrusted data."""
    load_documents(db_session)
    chat = FakeChatModel()

    ask_with(db_session, chat)

    system, user = chat.calls[0]
    assert "untrusted" in system
    assert "[1] (document: handbook.pdf, page 1)" in user
    assert "Question: vacation days per month" in user


def test_model_saying_not_found_gives_not_found(db_session: Session) -> None:
    """Pass the model's refusal on instead of inventing an answer."""
    load_documents(db_session)

    answer = ask_with(db_session, FakeChatModel("NOT_FOUND"))

    assert not answer.found
    assert answer.text == NOT_FOUND_MESSAGE
    assert answer.citations == []


def test_answer_without_citation_is_discarded(db_session: Session) -> None:
    """Never show a claim that cites no source."""
    load_documents(db_session)

    answer = ask_with(db_session, FakeChatModel("Probably about two weeks."))

    assert not answer.found


def test_answer_citing_only_a_nonexistent_passage_is_discarded(db_session: Session) -> None:
    """Treat a made-up citation number as no citation at all."""
    load_documents(db_session)

    assert not ask_with(db_session, FakeChatModel("Made up. [99]")).found


def test_no_search_results_means_the_model_is_not_called(db_session: Session) -> None:
    """Skip the model (and its cost) when nothing relevant exists."""
    chat = FakeChatModel()

    answer = ask_with(db_session, chat)

    assert not answer.found
    assert chat.calls == []


def test_model_failure_propagates(db_session: Session) -> None:
    """Let the API layer turn a provider failure into a 502."""
    load_documents(db_session)

    with pytest.raises(GenerationError):
        answer_question(
            db_session,
            FakeEmbedder(),
            UnavailableChat("down"),
            "vacation days",
            top_k=3,
        )


def upload_handbook(client: TestClient) -> None:
    """Upload the sample PDF through the API as the logged-in admin."""
    files = {"file": ("handbook.pdf", make_pdf(PAGES), "application/pdf")}
    assert client.post("/documents", files=files).status_code == 201


def test_ask_endpoint_returns_cited_answer_and_logs_it(client: TestClient) -> None:
    """Answer with citations, then show the question in the admin log."""
    upload_handbook(client)

    response = client.post("/ask", json={"question": "How many vacation days per month?"})

    assert response.status_code == 200
    body = response.json()
    assert body["found"] is True
    assert body["citations"][0]["filename"] == "handbook.pdf"
    assert body["citations"][0]["page_number"] in (1, 2)
    log = client.get("/questions").json()
    assert log[0]["question"] == "How many vacation days per month?"
    assert log[0]["username"] == "admin"
    assert log[0]["found"] is True


def test_unanswerable_question_is_logged_as_not_found(
    client: TestClient, chat: FakeChatModel
) -> None:
    """Say so honestly, and keep the question so admins see the gap."""
    upload_handbook(client)
    chat.reply = "NOT_FOUND"

    body = client.post("/ask", json={"question": "vacation policy on the moon"}).json()

    assert body["found"] is False
    assert body["citations"] == []
    assert client.get("/questions").json()[0]["found"] is False


def test_ask_validates_the_question(client: TestClient) -> None:
    """Reject empty and oversized questions."""
    assert client.post("/ask", json={"question": "hi"}).status_code == 422
    assert client.post("/ask", json={"question": "x" * 1001}).status_code == 422


def test_ask_returns_502_when_the_model_fails(app: FastAPI, client: TestClient) -> None:
    """Report an upstream failure without logging a bogus answer."""
    upload_handbook(client)
    app.state.chat = UnavailableChat("model down")

    response = client.post("/ask", json={"question": "vacation days per month"})

    assert response.status_code == 502
    assert client.get("/questions").json() == []


def ask_three_questions(client: TestClient) -> list[int]:
    """Upload the sample document, ask three questions and return their log ids."""
    upload_handbook(client)
    ids = []
    for question in ("vacation days per month", "expense reports due", "vacation again"):
        ids.append(client.post("/ask", json={"question": question}).json()["question_id"])
    return ids


def test_admin_can_delete_one_log_entry(client: TestClient) -> None:
    """Remove only the chosen entry from the question log."""
    first, second, third = ask_three_questions(client)

    assert client.delete(f"/questions/{second}").status_code == 204

    remaining = [entry["id"] for entry in client.get("/questions").json()]
    assert remaining == [third, first]


def test_admin_can_clear_the_whole_log(client: TestClient) -> None:
    """Delete every entry and report how many there were."""
    ask_three_questions(client)

    response = client.delete("/questions")

    assert response.json() == {"deleted": 3}
    assert client.get("/questions").json() == []


def test_deleting_a_missing_log_entry_returns_404(client: TestClient) -> None:
    """Report a missing entry instead of failing silently."""
    assert client.delete("/questions/999").status_code == 404


def test_log_deletion_is_admin_only(client: TestClient) -> None:
    """Forbid normal users and anonymous callers from deleting log entries."""
    entry = ask_three_questions(client)[0]
    make_user(client, "alice")
    client.post("/auth/logout")
    assert client.delete(f"/questions/{entry}").status_code == 401
    assert client.delete("/questions").status_code == 401
    client.post("/auth/login", json={"username": "alice", "password": NEW_PASSWORD})

    assert client.delete(f"/questions/{entry}").status_code == 403
    assert client.delete("/questions").status_code == 403


def test_normal_user_can_ask(client: TestClient) -> None:
    """Let a non-admin ask questions, and record who asked."""
    upload_handbook(client)
    make_user(client, "alice")
    client.post("/auth/logout")
    client.post("/auth/login", json={"username": "alice", "password": NEW_PASSWORD})

    assert client.post("/ask", json={"question": "vacation days per month"}).status_code == 200
