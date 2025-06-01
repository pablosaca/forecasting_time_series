import logging


def get_logger():
    logger = logging.getLogger("Endesa - Consumo Gas")
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:  # Evita añadir múltiples handlers si se importa varias veces
        formatter = logging.Formatter(
            '%(asctime)s | %(name)s | %(levelname)s | %(module)s | [Línea: %(lineno)d] | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger
