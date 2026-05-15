import asyncio
import ctypes.util
import logging
from typing import Any, Optional

from config.config_loader import load_app_config
from core.dispatcher import Dispatcher
from core.exceptions import EngineStop
from core.startup_diagnostics import ModuleSeverity, StartupDiagnostics
from hooks.manager import HookManager
from utils.logging_config import setup_logging

# Патч ctypes.find_library для Whisper на Windows
_orig_find = ctypes.util.find_library


def _patched_find_library(name: str) -> Optional[str]:
    if name == "c":
        return "msvcrt"
    return _orig_find(name)


ctypes.util.find_library = _patched_find_library

try:
    from interaction.whisper_input import ImprovedWhisperASR
except ImportError:
    ImprovedWhisperASR = None

try:
    from interaction.pyttsx3_output import Pyttsx3TTS
except ImportError:
    Pyttsx3TTS = None

try:
    from interaction.edge_tts_output import EdgeTTS
except ImportError:
    EdgeTTS = None


class KeyboardFallbackASR:
    """Fallback ASR — клавиатура. Используется когда Whisper не загрузился."""

    fallback_mode = True

    async def listen(self) -> str:
        loop = asyncio.get_running_loop()
        text = await loop.run_in_executor(None, lambda: input("Введите команду: "))
        return text.strip()


class ConsoleTTS:
    """Fallback TTS — печать в консоль. Используется когда pyttsx3 не загрузился."""

    fallback_mode = True

    async def say(self, text: str) -> None:
        if text and text.strip():
            print(f"Redmond: {text}")


class RedmondEngine:
    """Основной цикл: ASR → Dispatcher → TTS. С graceful-fallback и diagnostics."""

    def __init__(
        self,
        config: Any = None,
        asr: Any = None,
        tts: Any = None,
        diagnostics: Optional[StartupDiagnostics] = None,
    ):
        self.config = config or load_app_config()
        self.diagnostics = diagnostics or StartupDiagnostics()
        setup_logging(self.config.log_level)
        self.logger = logging.getLogger(self.__class__.__name__)

        self.asr = self._init_asr(asr)
        self.tts = self._init_tts(tts)

        try:
            self.dispatcher = Dispatcher(self.config)
            self.dispatcher.bind_tts(self.tts)
            self._report_dispatcher_health()
        except Exception as e:
            self.diagnostics.error("Dispatcher", str(e), severity=ModuleSeverity.CRITICAL)
            raise

        try:
            self.hooks = HookManager()
            self.diagnostics.ok("Hooks")
        except Exception as e:
            self.diagnostics.error("Hooks", str(e), severity=ModuleSeverity.CRITICAL)
            raise

        self.diagnostics.ok("Engine loop", "ready", severity=ModuleSeverity.CRITICAL)

    # ---------- инициализация компонентов ----------

    def _init_asr(self, provided: Any) -> Any:
        if provided is not None:
            self.diagnostics.ok("Whisper ASR", "provided externally")
            return provided

        if ImprovedWhisperASR is None:
            self.diagnostics.warn("Whisper ASR", "fallback: keyboard input (whisper unavailable)")
            return KeyboardFallbackASR()

        try:
            self.logger.info("Initializing ImprovedWhisperASR on %s", self.config.whisper_device)
            asr = ImprovedWhisperASR(
                model_size=self.config.whisper_model_size,
                device=self.config.whisper_device,
                language=None,  # auto-detect (mультиязычный режим)
            )
            self.diagnostics.ok(
                "Whisper ASR",
                f"model={self.config.whisper_model_size}, device={self.config.whisper_device}, lang=auto",
            )
            return asr
        except Exception as e:
            self.logger.exception("ASR init failed — keyboard fallback")
            self.diagnostics.warn("Whisper ASR", f"fallback: keyboard ({e})")
            return KeyboardFallbackASR()

    def _init_tts(self, provided: Any) -> Any:
        if provided is not None:
            self.diagnostics.ok("TTS", "provided externally")
            return provided

        engine_name = getattr(self.config, "tts_engine", "pyttsx3").lower()

        if engine_name == "edge-tts":
            return self._init_edge_tts()
        if engine_name == "pyttsx3":
            return self._init_pyttsx3()
        if engine_name == "console":
            self.diagnostics.warn("TTS (console)", "stdout output")
            return ConsoleTTS()

        self.diagnostics.warn("TTS", f"unknown engine '{engine_name}' — pyttsx3 fallback")
        return self._init_pyttsx3()

    def _init_edge_tts(self) -> Any:
        if EdgeTTS is None:
            self.diagnostics.warn("TTS (edge-tts)", "package not installed — pyttsx3 fallback")
            return self._init_pyttsx3()
        try:
            voice = getattr(self.config, "edge_tts_voice", "en-US-BrianMultilingualNeural")
            rate = getattr(self.config, "edge_tts_rate", "+0%")
            tts = EdgeTTS(voice=voice, rate=rate)
            if getattr(tts, "fallback_mode", False):
                self.diagnostics.warn("TTS (edge-tts)", "fallback: console output")
            else:
                self.diagnostics.ok("TTS (edge-tts)", f"voice={voice}")
            return tts
        except Exception as e:
            self.logger.exception("EdgeTTS init failed — pyttsx3 fallback")
            self.diagnostics.warn("TTS (edge-tts)", f"fallback ({e})")
            return self._init_pyttsx3()

    def _init_pyttsx3(self) -> Any:
        if Pyttsx3TTS is None:
            self.diagnostics.warn("TTS (pyttsx3)", "fallback: console output")
            return ConsoleTTS()
        try:
            tts = Pyttsx3TTS()
            if getattr(tts, "fallback_mode", False):
                self.diagnostics.warn("TTS (pyttsx3)", "fallback: console output")
            else:
                self.diagnostics.ok("TTS (pyttsx3)")
            return tts
        except Exception as e:
            self.logger.exception("pyttsx3 init failed — console fallback")
            self.diagnostics.warn("TTS (pyttsx3)", f"fallback: console ({e})")
            return ConsoleTTS()

    def _report_dispatcher_health(self) -> None:
        if self.dispatcher.auth is not None:
            self.diagnostics.ok("Auth")
        if self.dispatcher.safety is not None:
            self.diagnostics.ok("Safety/GoalManager")

        rg = self.dispatcher.response_generator
        if rg is None:
            self.diagnostics.error("ResponseGenerator", "not initialized", severity=ModuleSeverity.CRITICAL)
            return

        if rg.mem is not None:
            self.diagnostics.ok("MemoryStore", f"db={rg.mem.path}")
        else:
            self.diagnostics.warn("MemoryStore", "disabled")

        if rg.searcher is not None:
            missing = []
            if not getattr(rg.searcher, "api_key", ""):
                missing.append("API key")
            if not getattr(rg.searcher, "cx", ""):
                missing.append("search engine id")
            if missing:
                self.diagnostics.warn("Web Search (Google)", "missing " + ", ".join(missing))
            else:
                self.diagnostics.ok("Web Search (Google)")
        else:
            self.diagnostics.warn("Web Search (Google)", "disabled")

        self._report_provider_health(rg)

    def _report_provider_health(self, rg) -> None:
        providers = set(getattr(self.config, "llm_provider_order", []))

        if "groq" in providers:
            if getattr(self.config, "groq_api_key", ""):
                self.diagnostics.ok("Groq Provider", f"model={self.config.groq_model}")
            else:
                self.diagnostics.warn("Groq Provider", "missing API key")
        if "ollama" in providers:
            self.diagnostics.ok("Ollama Provider", f"url={self.config.ollama_base_url}")
        if "gemini" in providers:
            if getattr(self.config, "gemini_api_key", ""):
                self.diagnostics.ok("Gemini Provider", f"model={self.config.gemini_model}")
            else:
                self.diagnostics.warn("Gemini Provider", "missing API key")
        if "transformers" in providers:
            if rg.model is not None:
                self.diagnostics.ok("Transformers Provider", self.config.llm_model_path)
            else:
                self.diagnostics.warn("Transformers Provider", "model unavailable")

    # ---------- основной цикл ----------

    def run(self) -> None:
        try:
            asyncio.run(self.start())
        except KeyboardInterrupt:
            self.logger.info("Завершение по Ctrl+C")
        except EngineStop:
            self.logger.info("EngineStop")

    async def start(self) -> None:
        self.diagnostics.summary()
        self.logger.info("RedmondEngine стартует")

        await self.hooks.run("on_start")
        await self.tts.say("Привет! Я Redmond.")

        try:
            await self._input_loop()
        finally:
            await self.hooks.run("on_stop")

    async def _input_loop(self) -> None:
        while True:
            try:
                text = await asyncio.wait_for(
                    self.asr.listen(),
                    timeout=self.config.idle_timeout_sec,
                )
            except asyncio.TimeoutError:
                continue
            except EngineStop:
                raise
            except Exception:
                self.logger.exception("Ошибка ASR.listen()")
                continue

            try:
                await self.dispatcher.dispatch(text)
            except EngineStop:
                raise
            except Exception:
                self.logger.exception("Ошибка dispatcher.dispatch()")
