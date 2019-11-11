import os

class BrownWrapper:

    outfile = None
    filename = "input.txt"

    def __init__(self, data):

        f = open(self.filename, "w")
        for doc in data:
            for sen in doc:
                f.write(sen[0] + " ")
            f.write("\n")
        f.close()
        outdir = os.popen("./brown/wcluster --text " + self.filename + " --c 5").read()
        outlogfile = outdir.split(" ")[2]
        outdir = outlogfile.split("/")[0]
        self.outfile = outdir + "/paths"

    def get_brown_clustering(self):
        brown_dict = dict()
        with open(self.outfile) as f:
            content = f.readlines()
            for line in content:
                words = line.split("\t")
                brown_dict.update({words[1]: (words[0], words[2].strip())})
        return brown_dict
