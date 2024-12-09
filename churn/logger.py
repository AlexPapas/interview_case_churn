import logging

logger = logging.getLogger(__name__)

# Determines where the logs go to:
handler = logging.StreamHandler()

# Determines what the logs will look like:
fmt = "%(levelname)s [ %(asctime)s | %(filename)s | %(funcName)s ] %(message)s"
formatter = logging.Formatter(fmt)
handler.setFormatter(formatter)

logger.addHandler(handler)
