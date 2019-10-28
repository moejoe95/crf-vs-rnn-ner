# parser for reuters.xml, not used at the moment

import codecs
from bs4.element import Tag
from bs4 import BeautifulSoup as bs

def parse(datafile, textTag, namedEntityTag):
    # Read data file and parse the XML
    with codecs.open(datafile, "r", "utf-8") as infile:
        soup = bs(infile, "html5lib")

    docs = []
    for elem in soup.find_all("document"):
        texts = []

        for c in elem.find(textTag).children:
            if type(c) == Tag:
                if c.name == namedEntityTag:
                    label = "N"  # part of a named entity
                else:
                    label = "I"  # irrelevant word
                for w in c.text.split(" "):
                    if len(w) > 0:
                        texts.append((w, label))
        docs.append(texts)
    return docs