from gazegraph.training.tasks.base_task import BaseTask
from gazegraph.training.tasks.next_action import NextActionTask
from gazegraph.training.tasks.future_actions import FutureActionsTask

__all__ = [
    'BaseTask',
    'NextActionTask',
    'FutureActionsTask',
]

def get_task(task_name):
    task_map = {
        "future_actions": FutureActionsTask,
        "next_action": NextActionTask,
    }
    return task_map[task_name]