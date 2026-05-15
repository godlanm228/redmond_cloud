async def on_start_hook(engine):
    await engine.tts.say("Привет! Я Redmond. Чем займёмся сегодня?")

def register(manager, engine):
    # manager.register(<имя события>, <корутина без аргументов>)
    manager.register("on_start", lambda: on_start_hook(engine))