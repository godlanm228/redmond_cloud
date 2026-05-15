"""Аналитические интенты: отчёты, анализ продаж."""

from typing import Optional

from handlers import register


@register("show_report")
def handle_show_report(ctx) -> Optional[str]:
    if ctx.user_role == "guest":
        return "Отчёты доступны только авторизованным пользователям."

    rg = ctx.rg
    if rg is None:
        return "Отчёт недоступен (response_generator не подключен)."

    total_interactions = len(rg.history)
    total_memory = rg.mem.count() if rg.mem else 0

    providers = []
    if getattr(rg.config, "groq_api_key", ""):
        providers.append("Groq")
    if getattr(rg.config, "gemini_api_key", ""):
        providers.append("Gemini")
    if rg.model is not None:
        providers.append("Transformers")

    return (
        "Отчёт системы:\n"
        f"- Взаимодействий: {total_interactions}\n"
        f"- Записей в памяти: {total_memory}\n"
        f"- LLM-провайдеры: {', '.join(providers) or 'нет'}\n"
        f"- Поиск: {'активен' if rg.searcher else 'недоступен'}"
    )


@register("analyze_sales")
def handle_analyze_sales(ctx) -> Optional[str]:
    if ctx.user_role == "guest":
        return "Анализ продаж доступен только авторизованным пользователям."
    return "Модуль анализа продаж в разработке. Используйте «показать отчёт» для базовой статистики."
