
from .future_actions import FutureActionsTask
from .next_action import NextActionTask

def get_task(task_name):
    task_map = {
        "future_actions": FutureActionsTask,
        "next_action": NextActionTask,
    }
    return task_map[task_name]