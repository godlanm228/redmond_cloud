"""Операции над памятью: пометка важной записи."""

from typing import Optional

from handlers import register


@register("mark_important")
def handle_mark_important(ctx) -> Optional[str]:
    rg = ctx.rg
    if rg is None or rg.mem is None:
        return "Память сейчас недоступна."
    last_id = rg.mem.count()
    if last_id <= 0:
        return "Нет записей для пометки."
    rg.mem.mark_important(last_id)
    return "Запись помечена как важная."
