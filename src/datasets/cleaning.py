"""
Library of various cleaning-related functions, regular expressions and variables.
"""
import re
import string


simple_latin = string.ascii_lowercase + string.ascii_uppercase
dirty_chars = string.digits + string.punctuation


def is_clean_text(text: str) -> bool:
    """
    Simple text cleaning method.
    """
    dirty = (
        len(text) < 25                                               # Short text
        or
        0.5 < sum(char in dirty_chars for char in text) / len(text)  # More than 50% dirty chars                                            
    )
    return not dirty


# Source: https://gist.github.com/dperini/729294
url_regex = re.compile(
    r'(?:^|(?<![\w\/\.]))'
    r'(?:(?:https?:\/\/|ftp:\/\/|www\d{0,3}\.))'
    r'(?:\S+(?::\S*)?@)?' r'(?:'
    r'(?!(?:10|127)(?:\.\d{1,3}){3})'
    r'(?!(?:169\.254|192\.168)(?:\.\d{1,3}){2})'
    r'(?!172\.(?:1[6-9]|2\d|3[0-1])(?:\.\d{1,3}){2})'
    r'(?:[1-9]\d?|1\d\d|2[01]\d|22[0-3])'
    r'(?:\.(?:1?\d{1,2}|2[0-4]\d|25[0-5])){2}'
    r'(?:\.(?:[1-9]\d?|1\d\d|2[0-4]\d|25[0-4]))'
    r'|'
    r'(?:(?:[a-z\\u00a1-\\uffff0-9]-?)*[a-z\\u00a1-\\uffff0-9]+)'
    r'(?:\.(?:[a-z\\u00a1-\\uffff0-9]-?)*[a-z\\u00a1-\\uffff0-9]+)*'
    r'(?:\.(?:[a-z\\u00a1-\\uffff]{2,}))' r'|' r'(?:(localhost))' r')'
    r'(?::\d{2,5})?'
    r'(?:\/[^\)\]\}\s]*)?',
    flags=re.IGNORECASE,
)


def remove_urls(text: str) -> str:
    return url_regex.sub('', text)


# Source: https://gist.github.com/Nikitha2309/15337f4f593c4a21fb0965804755c41d
emoji_regex = re.compile('['
        u'\U0001F600-\U0001F64F'  # emoticons
        u'\U0001F300-\U0001F5FF'  # symbols & pictographs
        u'\U0001F680-\U0001F6FF'  # transport & map symbols
        u'\U0001F1E0-\U0001F1FF'  # flags (iOS)
        u'\U00002500-\U00002BEF'  # chinese char
        u'\U00002702-\U000027B0'
        u'\U00002702-\U000027B0'
        u'\U000024C2-\U0001F251'
        u'\U0001f926-\U0001f937'
        u'\U00010000-\U0010ffff'
        u'\u2640-\u2642'
        u'\u2600-\u2B55'
        u'\u200d'
        u'\u23cf'
        u'\u23e9'
        u'\u231a'
        u'\ufe0f'  # dingbats
        u'\u3030'
    ']+')


def remove_emojis(text: str) -> str:
    return emoji_regex.sub('', text)


sentence_stop_regex = re.compile('['
    u'\u002e' # full stop
    u'\u2026' # ellipsis
    u'\u061F' # arabic question mark
    u'\u06D4' # arabic full stop
    u'\u2022' # bullet point
    u'\u3002' # chinese period
    u'\u25CB' # white circle
    '\|'      # pipe
']+')


def replace_stops(text: str) -> str:
    """
    Replaces some characters that are being used to end sentences. Used for sentence segmentation with sliding windows.
    """
    return sentence_stop_regex.sub('.', text)


whitespace_regex = re.compile(r'\s+')


def replace_whitespaces(text: str) -> str:
    return whitespace_regex.sub(' ', text)


def clean_ocr(ocr: str) -> str:
    """
    Remove all lines that are shorter than 6 and have more than 50% `dirty_chars`.
    """
    return '\n'.join(
        line
        for line in ocr.split('\n')
        if len(line) > 5 and sum(char in dirty_chars for char in line) / len(line) < 0.5
    )


def clean_twitter_picture_links(text):
    """
    Replaces links to picture in twitter post only with 'pic'. 
    """
    return re.sub(r'pic.twitter.com/\S+', 'pic', text)


def clean_twitter_links(text):
    """
    Replaces twitter links with 't.co'.
    """
    return re.sub(r'\S+//t.co/\S+', 't.co', text)


def remove_elongation(text):
    """
    Replaces any occurrence of a string of consecutive identical non-space 
    characters (at least three in a row) with just one instance of that character.
    """
    text = re.sub(r'(\S+)\1{2,}', r'\1', text)
    return text