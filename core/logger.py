#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Logger - Sistema de logging persistente para OpenTopologyC

Guarda logs a archivo opentopologyc.log y muestra en consola.
"""

import logging
from pathlib import Path
from datetime import datetime


def setup_logger(
    name: str = "opentopologyc",
    log_file: str = "opentopologyc.log",
    level: int = logging.INFO,
    console: bool = True
) -> logging.Logger:
    """
    Configura logger con archivo y opcionalmente consola

    Args:
        name: Nombre del logger
        log_file: Ruta al archivo de log
        level: Nivel de logging (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        console: Si True, también muestra en consola

    Returns:
        Logger configurado
    """
    logger = logging.getLogger(name)

    # Evitar duplicar handlers si ya existe
    if logger.handlers:
        return logger

    logger.setLevel(level)

    # Formato de log
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Handler de archivo
    log_path = Path(log_file)
    file_handler = logging.FileHandler(log_path, encoding='utf-8')
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Handler de consola (opcional)
    if console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger


def log_session_start(logger: logging.Logger, module: str):
    """Registra inicio de sesión"""
    logger.info("="*60)
    logger.info(f"SESIÓN INICIADA - {module}")
    logger.info(f"Timestamp: {datetime.now().isoformat()}")
    logger.info("="*60)


def log_session_end(logger: logging.Logger):
    """Registra fin de sesión"""
    logger.info("="*60)
    logger.info("SESIÓN FINALIZADA")
    logger.info("="*60)
    logger.info("")  # Línea en blanco para separar sesiones


# Logger global por defecto
_default_logger = None


def get_logger() -> logging.Logger:
    """Obtiene el logger global por defecto"""
    global _default_logger
    if _default_logger is None:
        _default_logger = setup_logger()
    return _default_logger


# Ejemplo de uso
if __name__ == "__main__":
    # Crear logger
    logger = setup_logger(log_file="test.log")

    # Registrar mensajes
    log_session_start(logger, "TEST")

    logger.debug("Mensaje de debug")
    logger.info("Mensaje informativo")
    logger.warning("Advertencia")
    logger.error("Error")
    logger.critical("Error crítico")

    log_session_end(logger)

    print("\n✓ Log guardado en test.log")
