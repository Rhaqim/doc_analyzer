# Create a logger instance
import logging

logger = logging.getLogger(__name__)

# Set the logging level (you can adjust this as needed)
logger.setLevel(logging.DEBUG)

# Create a FileHandler to log messages to a file
file_handler = logging.FileHandler("app.log")

# Create a StreamHandler to log messages to the console
stream_handler = logging.StreamHandler()

# Create a Formatter to format log messages
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Set the formatter for the handlers
file_handler.setFormatter(formatter)
stream_handler.setFormatter(formatter)

# Add the handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(stream_handler)
