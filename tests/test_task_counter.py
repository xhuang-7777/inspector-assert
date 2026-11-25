from sample_code.task_counter import count_completed


def test_empty_task_list():
    assert count_completed([]) == 0


def test_mixed_task_list():
    tasks = [{"done": True}, {"done": False}, {"done": True}]
    assert count_completed(tasks) == 2


def test_incomplete_tasks_not_counted():
    assert count_completed([{"done": False}]) == 0
