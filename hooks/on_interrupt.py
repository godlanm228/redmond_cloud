async def on_interrupt_hook(engine):
    engine.logger.debug("Сработал on_interrupt")

def register(manager, engine):
    manager.register("on_interrupt", lambda: on_interrupt_hook(engine))