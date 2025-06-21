from gazegraph.training.tasks.base_task import BaseTask
from gazegraph.training.tasks.next_action import NextActionTask
from gazegraph.training.tasks.future_actions import FutureActionsTask
from gazegraph.training.tasks.action_recognition import ActionRecognitionTask
from gazegraph.training.tasks.object_recognition import ObjectRecognitionTask

__all__ = [
    "BaseTask",
    "NextActionTask",
    "FutureActionsTask",
    "ActionRecognitionTask",
    "ObjectRecognitionTask",
]


def get_task(task_name):
    task_map = {
        "future_actions": FutureActionsTask,
        "next_action": NextActionTask,
        "action_recognition": ActionRecognitionTask,
        "object_recognition": ObjectRecognitionTask,
    }
    return task_map[task_name]
