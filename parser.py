# parse CATH database and extract the domain boundary position information of each sequence. (can also extract fragment information)
import csv
import re

class Seq:
    def __init__(self):
        self.name = ""
        self.dNum = 0 # domain
        self.fNum = 0 # fragment
        self.domains = []
        self.fragments = []

    def _parseLine_(self, line):
        self.name = line[0:5]
        # 6 space 7 D
        self.dNum = int(line[8:10])
        # 10 space 11 F
        self.fNum = int(line[12:14])

        p = 0 # position in rest of the string
        r = line[14:]
        dashCount = 0
        dom_sign = re.compile(r"^[0-9]\s{2}[A-Z0-9]$")
        while p < len(r):
            if len(self.domains) < self.dNum \
            and re.match(dom_sign,r[p:p+4]) != None:
                domain = Domain() # create new domain
                dashCount = 0
                domain.sNum = r[p]
                doStr = ""
                # use segment number to be a condition
                while dashCount < 2 * int(domain.sNum):
                    if r[p] == "-":
                        dashCount += 1
                    doStr += r[p]
                    # print(doStr)
                    # print("dashCount:" + " " + str(dashCount))
                    p +=1
                domain._parse_(doStr)
                self.domains.append(domain)
            # read fragment data
            # elif len(self.fragments) < self.fNum and r[p].isalpha():
            #     frag = Fragment()
            #     fragStr = ""
            #     dashCount=0
            #     while dashCount < 2:
            #         if r[p] == "-":
            #             dashCount +=1
            #         fragStr += r[p]
            #         p+=1
            #     frag._parse_(fragStr)
            #     self.fragments.append(frag)
            else:
                p+=1
        self._show_()

    def _show_(self):

        return(  self.name + " " + \
                str(self.dNum))

        for d in self.domains:
            d.show()

        for f in self.fragments:
            f.show()

    def __getitem__(self,):

        return self.name

    def _toString_(self):
        return str(self.domains)


class Domain:
    def __init__(self):
        self.sNum = 0 # segment num
        self.start = 0
        self.end = 0

    def _parse_(self, str):
        info = str.split()
        # print(info)
        self.sNum = info[0]
        self.start = info[2]
        self.end = info[-2]

    def show(self):
        return self.start + " " + self.end

class Fragment:
    def _init_(self):
        self.chainChar = ""
        self.start = 0
        self.end = 0
        self.rNum = 0

    def _parse_(self, str):
        info = str.split()
        print(info)
        # self.sNum = info[1]
        self.start = info[1]
        self.rNum = info[-1][info[-1].find("(")+1:info[-1].find(")")]
        self.end = info[-2]

    def show(self):
        print(str(self.start) + " " + str(self.end) + " " + str(self.rNum))


# read CATH file from specified path and output the sequence name and the corresponding domain boundary position information
def read_CATH_Dom(CATH_file_path,output_path):
    try:
        # seq dictionary
        seqs = {}

        with open(CATH_file_path) as domBound:
        # with open() as f:
            lines = domBound.readlines()
            i= 0
            for line in lines:
                i +=1
                print(i)
                seq = Seq()
                seq._parseLine_(line)
                seqs[seq.name]=[domain.show().split() for domain in seq.domains]
        keyList = seqs.keys()
        valueList = seqs.values()

        rows = zip(keyList, valueList)

        with open(output_path,"wb") as f:
            w = csv.writer(f)
            for row in rows:
                w.writerow(row)
    except OSError:
        print("the input CATH file path is not available")

if __name__ == "__main__":
    CATH_file_path = "/Users/Graceyh/Google Drive/AboutDissertation/Data/CathDomall.v4.0.0"
    test_file_path = "/Users/Graceyh/Google Drive/AboutDissertation/Data/test.txt"
    output_path = "/Users/Graceyh/Google Drive/AboutDissertation/Data/DomBound(con).csv"

    read_CATH_Dom(CATH_file_path, output_path)
