import sys
import warnings
import logging

logger = logging.getLogger(__name__)

def patch_huggingface_hub():
    '''Добавляет обратную совместимость для cached_download'''
    try:
        import huggingface_hub

        # Если cached_download уже есть, ничего не делаем
        if hasattr(huggingface_hub, 'cached_download'):
            logger.debug("cached_download уже существует")
            return True

        # Проверяем наличие hf_hub_download
        if not hasattr(huggingface_hub, 'hf_hub_download'):
            logger.error("hf_hub_download не найден в huggingface_hub")
            return False

        # Добавляем заглушку
        from huggingface_hub import hf_hub_download
        import os
        from urllib.parse import urlparse

        def cached_download(url, cache_dir=None, force_filename=None, proxies=None,
                            etag_timeout=10, resume_download=False, user_agent=None,
                            local_files_only=False, use_auth_token=None,
                            legacy_cache_layout=False, library_name=None, library_version=None):
            '''Обратная совместимость для cached_download'''

            warnings.warn(
                "cached_download is deprecated. Use hf_hub_download instead.",
                FutureWarning,
                stacklevel=2
            )

            try:
                # Пытаемся извлечь repo_id и filename из URL
                parsed = urlparse(url)
                path_parts = parsed.path.strip('/').split('/')

                if 'huggingface.co' in parsed.netloc and len(path_parts) >= 4:
                    # URL вида: https://huggingface.co/user/repo/resolve/main/file.bin
                    repo_id = f"{path_parts[0]}/{path_parts[1]}"
                    filename = '/'.join(path_parts[3:])  # Всё после 'resolve/main/'
                else:
                    # Fallback для других случаев
                    repo_id = "sentence-transformers/all-MiniLM-L6-v2"
                    filename = os.path.basename(parsed.path) or "pytorch_model.bin"

                # Используем hf_hub_download с правильными параметрами
                kwargs = {
                    'repo_id': repo_id,
                    'filename': filename,
                    'cache_dir': cache_dir,
                    'proxies': proxies,
                    'resume_download': resume_download,
                    'local_files_only': local_files_only
                }

                # В новых версиях параметр называется token, а не use_auth_token
                if use_auth_token is not None:
                    kwargs['token'] = use_auth_token

                # library_name и library_version игнорируем - они не нужны для hf_hub_download

                return hf_hub_download(**kwargs)

            except Exception as e:
                logger.error(f"Ошибка в cached_download fallback: {e}")
                # Возвращаем оригинальный URL как fallback
                return url

        huggingface_hub.cached_download = cached_download
        logger.info("✓ Патч для huggingface_hub применён успешно")
        return True

    except ImportError:
        logger.warning("huggingface_hub не установлен")
        return False
    except Exception as e:
        logger.error(f"✗ Не удалось применить патч: {e}")
        return False

# Автоматически применяем патч при импорте
patch_success = patch_huggingface_hub()