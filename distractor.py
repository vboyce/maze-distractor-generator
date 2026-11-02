class distractor:
    """General class for distractor words"""

    def __init__(self, text, freq):
        self.text = text
        self.len = len(text)
        self.freq = freq

