"""Entry point for the llm_proxy ChatML server."""

import logging
import os
import signal
from datetime import datetime

import httpx
import uvicorn

from llmdataproxy.config import parse_args, _CONFIG_FIELDS
from llmdataproxy.server import create_app
from llmdataproxy.session import SessionManager


def setup_logging(log_folder):
    os.makedirs(log_folder, exist_ok=True)
    log_name = "llm_proxy_" + datetime.now().strftime("%m%d_%H%M") + ".log"
    log_path = os.path.join(log_folder, log_name)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger("llm_proxy")


def main():
    args = parse_args()
    logger = setup_logging(args.log_folder)

    # --- log resolved config, marking deviations from DEFAULT ---
    base = getattr(args, "_defaults", {})
    parts = []
    for key in _CONFIG_FIELDS:
        val = getattr(args, key)
        if key in ("api_key",) and val:
            display = val[:8] + "***" if len(val) > 8 else "***"
        else:
            display = val
        marker = "*" if base and val != base.get(key) else " "
        parts.append(f"{marker}{key}={display}")
    logger.info("config: %s", "  ".join(parts))

    session_mgr = SessionManager(args.log_folder, args.session_name, args.log_chatml,
                                 args.session_path, rl_enabled=args.rl)

    # --- fetch available models from upstream ---
    available_models = []
    base_url = args.base_url.rstrip("/")
    for attempt, suffix in enumerate(("", "/v1")):
        try:
            models_url = f"{base_url}{suffix}/models"
            resp = httpx.get(models_url,
                             headers={"Authorization": f"Bearer {args.api_key}"} if args.api_key else {},
                             timeout=10)
            resp.raise_for_status()
            model_list = resp.json().get("data", [])
            available_models = [m["id"] for m in model_list if m.get("id")]
            if suffix == "/v1":
                args.base_url = base_url + "/v1"
                logger.warning("auto-detected /v1 prefix: base-url updated to '%s'", args.base_url)
            logger.info("available models from upstream (%d): %s",
                        len(available_models), ", ".join(available_models))
            break
        except Exception as e:
            if attempt == 0:
                logger.debug("models fetch without /v1 failed, retrying with /v1: %s", e)
            else:
                logger.warning("failed to fetch models from upstream: %s", e)

    # --- default model ---
    default_model = args.default_model
    if default_model is None and available_models:
        default_model = available_models[0]
        logger.info("default-model auto-set to '%s' (first available)", default_model)
    elif default_model is not None:
        logger.info("default-model set to '%s' (from config)", default_model)

    app = create_app(args.base_url, args.api_key, session_mgr, args.temperature,
                     default_model=default_model, override_model=args.override_model)
    app.state.available_models = available_models

    # Graceful shutdown — dump sessions
    def shutdown():
        logger.info("shutting down, dumping ChatML sessions…")
        path = session_mgr.dump_all()
        logger.info("sessions dumped to %s", path)

    def _sig_handler(signum, frame):
        shutdown()
        os._exit(0)

    signal.signal(signal.SIGINT, _sig_handler)
    signal.signal(signal.SIGTERM, _sig_handler)

    logger.info("starting llm_proxy on %s:%d, upstream=%s, log_chatml=%s, log_folder=%s",
                args.host, args.port, args.base_url, args.log_chatml, args.log_folder)
    print(f"Initial session name: {args.session_name}")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info",
                access_log=False)


if __name__ == "__main__":
    main()
