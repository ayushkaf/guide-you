# Dialogue Module

This package handles intent parsing, dialogue management, and threaded processing for user-assistant conversation in the Visual Guidance Assistant.

- `intent_parser.py`: Parses user speech/text into structured intents.
- `manager.py`: Maps intents to responses, manages dialogue state.
- `dialogue_processor.py`: Threaded processor for handling dialogue asynchronously.

## Usage

Instantiate `DialogueProcessor` with command and response queues, config, and (optionally) context. Start the thread to enable real-time dialogue handling.