async def on_idle_hook(engine):
    await engine.tts.say("Похоже, вы бездействуете. Чем займёмся?")

def register(manager, engine):
    manager.register("on_idle", lambda: on_idle_hook(engine))