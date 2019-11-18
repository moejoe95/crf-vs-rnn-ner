import os

class BrownWrapper:

    outfile = None
    filename = "input.txt"
    clusters = '5'

    def __init__(self, data):

        f = open(self.filename, "w")
        for doc in data:
            for sen in doc:
                f.write(sen[0] + " ")
            f.write("\n")
        f.close()
        outdir = os.popen("./brown/wcluster --text " + self.filename + " --c " + self.clusters).read()
        outlogfile = outdir.split(" ")[2]
        outdir = outlogfile.split("/")[0]
        self.outfile = outdir + "/paths"


    def bin_to_dec(self, binary): 
        binary = int(binary)
        binary1 = binary
        decimal, i, n = 0, 0, 0
        while(binary != 0): 
            dec = binary % 10
            decimal = decimal + dec * pow(2, i) 
            binary = binary//10
            i += 1
        return decimal


    def get_brown_clustering(self):
        brown_dict = dict()
        with open(self.outfile) as f:
            content = f.readlines()
            for line in content:
                words = line.split("\t")
                brown_dict.update({words[1]: (words[0], words[2].strip(), self.bin_to_dec(words[0]))})
        return brown_dict
