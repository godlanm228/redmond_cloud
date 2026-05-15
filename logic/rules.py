class RuleManager:
    def __init__(self):
        self.limits = {}

    def apply(self, action, context):
        t = action.get("type")
        if t == "task":
            return "Добавил в план тренировку, сэр."
        elif t == "limit":
            self.limits[action["target"]] = action["duration"]
            return f"Понял. Ограничу {action['target']} до {action['duration']} минут."
        elif t == "memory":
            return "Запомнил, сэр."
        elif t == "block":
            return "Ограничение добавлено, сэр."
        return "Пока не знаю, что с этим делать."