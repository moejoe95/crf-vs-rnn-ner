import re


class WordShape:

    def __init__(self):
        """Instantiates a new object of this feature generator."""
        # maximum length of tokens after which to simply cut off
        self.max_length = 15
        # if cut off because of maximum length, use this char at the end of the word to signal
        # the cutoff
        self.max_length_char = "~"

        self.normalization = [
            (r"[A-ZÄÖÜ]", "A"),
            (r"[a-zäöüß]", "a"),
            (r"[0-9]", "9"),
            (r"[\.\!\?\,\;]", "."),
            (r"[\(\)\[\]\{\}]", "("),
            (r"[^Aa9\.\(]", "#")
        ]

        # note: we do not map numers to 9+, e.g. years will still be 9999
        self.mappings = [
            (r"[A]{2,}", "A+"),
            (r"[a]{2,}", "a+"),
            (r"[\.]{2,}", ".+"),
            (r"[\(]{2,}", "(+"),
            (r"[#]{2,}", "#+")
        ]

    def get_wordshape(self, token):
        """Converts a token/word to its word pattern.
        Args:
            token: The token/word to convert.
        Returns:
            The word pattern as string.
        """
        normalized = token
        for from_regex, to_str in self.normalization:
            normalized = re.sub(from_regex, to_str, normalized)

        wpattern = normalized
        for from_regex, to_str in self.mappings:
            wpattern = re.sub(from_regex, to_str, wpattern)

        if len(wpattern) > self.max_length:
            wpattern = wpattern[0:self.max_length] + self.max_length_char

        return wpattern