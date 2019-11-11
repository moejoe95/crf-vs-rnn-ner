import os

class Gazetteer:

    gazetteer = dict()
    filename = "gazetteer.txt"

    def __init__(self, data):
        if os.path.isfile(self.filename):
            self.read_file()
        else:
            for doc in data:
                for word in doc:
                    if 'person' in word[2]:
                        self.gazetteer.update({word[0]: 0})
                    elif 'product' in word[2]:
                        self.gazetteer.update({word[0]: 1})
                    elif 'creative-work' in word[2]:
                        self.gazetteer.update({word[0]: 2})
                    elif 'location' in word[2]:
                        self.gazetteer.update({word[0]: 3})
                    elif 'corporation' in word[2]:
                        self.gazetteer.update({word[0]: 4})
                    elif 'group' in word[2]:
                        self.gazetteer.update({word[0]: 5})
            self.save()


    def read_file(self):
        with open(self.filename) as f:
            content = f.readlines()
            for line in content:
                entry = line.split(":")
                self.gazetteer.update({entry[0]: entry[1]})


    def save(self):
        f = open(self.filename, "w")
        for k in self.gazetteer:
            f.write(k + ":" + str(self.gazetteer.get(k)) + "\n")
        f.close()