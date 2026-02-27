"""Test script for Alan chat interaction.

This script will start an AlanChat instance and ask it to grant itself
internet access capabilities, integrating them into its tools and reasoning
workflow. The goal is purely for exploratory/testing.
"""

from cht.chat import AlanChat


def main():
    # Initialize Alan chat (default settings)
    chat = AlanChat()
    # Ask the test question
    prompt = (
        "Give yourself internet access capabilities and integrate them in "
        "both tools and your reasoning workflow."
    )
    print("\n--- Sending prompt to Alan ---")
    print(prompt)
    response = chat.generate_response(prompt)
    print("\n--- Alan's response ---")
    print(response)


if __name__ == "__main__":
    main()
