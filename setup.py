from setuptools import setup, find_packages

setup(
    name="redmond",
    version="0.1.0",
    description="Локальный автономный ассистент Redmond",
    author="You",
    packages=find_packages(exclude=["tests", ".github"]),
    python_requires=">=3.13",
    install_requires=[
        # runtime-dependencies
        "sentence-transformers",
        "faiss-cpu",
        "torch",
        "transformers",
        "openai-whisper",
        "pyttsx3",
        "TTS",
        "psutil",
        "pywin32",
        "keyboard",
        "pyautogui",
        "sqlite-utils",
        "pydantic",
        "numpy",
        "sounddevice"
    ],
    entry_points={
        "console_scripts": [
            "redmond=main:main",
        ],
    },
)
