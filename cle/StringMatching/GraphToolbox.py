import re
from cle.StringMatching.InvertedIndexToolbox import InvertedIndex

class GraphManager:

    def __init__(self):
        self.graph = dict()
        self.indices = dict()

    def addNode(self, nodeid):
        if not self.containsNode(nodeid):
            self.graph[nodeid] = dict()

    def containsNode(self, nodeid):
        return nodeid in self.graph.keys()

    def addNodeProperty(self, nodeid, prop, value):
        if not self.containsNode(nodeid):
            self.addNode(nodeid)
        if not self.containsNodeProperty(nodeid, prop):
            self.graph[nodeid][prop] = value
            if prop in self.indices.keys():
                self.indices[prop].addToIndex(value, nodeid)

    def containsNodeProperty(self, nodeid, prop):
        if not self.containsNode(nodeid):
            return False
        return prop in self.graph[nodeid].keys()

    def readGraphFromNTFile(self, file_path):
        TEXT_REGEXER = re.compile("\".*\"")
        NODEID_REGEXER = re.compile("^<[^<^>]*>")
        PROPERTY_REGEXER = re.compile(" <[^<^>]*> ")
        file = open(file_path, "r", encoding="UTF-8")
        for line in file:
            try:
                value = TEXT_REGEXER.findall(line)[0].replace("\"", "")
            except:
                continue
            nodeid = NODEID_REGEXER.findall(line)[0].replace("\"", "")
            prop = PROPERTY_REGEXER.findall(line)[0].replace(" ", "")
            self.addNodeProperty(nodeid, prop, value)

    def setIndexProperties(self, props):
        for prop in props:
            self.indices[prop] = InvertedIndex()

