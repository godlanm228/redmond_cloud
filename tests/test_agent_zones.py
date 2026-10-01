"""Only Iris changes the owner's data; Redmond reads and hands over.

01.10.2026: Redmond had every tool. On «продли расписание» he extended it
himself, in Iris's voice («Продлила…») and around her rules — two agents
writing into one place. Specialisation by tools, not only by prompt.
"""

from logic import agents, toolbox

WRITES = {"add_diary_entry", "delete_diary_entry", "log_meal", "update_pantry",
          "add_goal", "mark_goal_done", "add_deadline", "mark_deadline_done",
          "delete_deadline", "postpone_deadline", "save_work_shift", "set_work_shift_status",
          "resolve_shift_conflict", "add_schedule_event", "remove_schedule_event",
          "extend_schedule", "stop_schedule_extension", "save_week_plan",
          "apply_file_items", "undo_file_items", "update_profile"}


def _offered(agent):
    return toolbox.legacy_names(toolbox.model_tools(agent.allowed_tools))


def test_redmond_can_read_but_not_change_the_owners_data():
    redmond = _offered(agents.agent_by_name("Redmond"))
    assert not (redmond & WRITES), f"Redmond может менять данные: {redmond & WRITES}"
    assert {"read_diary", "list_deadlines", "get_week_schedule", "ask_iris"} <= redmond


def test_iris_has_every_write():
    iris = _offered(agents.agent_by_name("Iris"))
    assert WRITES <= iris, f"Iris не хватает: {WRITES - iris}"


def test_content_free_messages_skip_the_reading():
    """Реальные сообщения владельца, где шаг понимания — лишний вызов."""
    from logic.agent_router import content_free
    for t in ("привет, ты тут?", "Живой?", "пасиба", "Продолжи", "Супер", ")", "Да", "лол"):
        assert content_free(t), t
    for t in ("сдал тест уже", "поел поел )", "что записала", "какая погода", "да, до 12 февраля"):
        assert not content_free(t), t
