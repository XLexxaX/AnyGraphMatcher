from collections import defaultdict
from datetime import timedelta, date



#http://rdflib.readthedocs.io/en/stable/_modules/rdflib/term.html#Literal     see XSDToPython
#from isodate import parse_time, parse_date, parse_datetime

def daterange(start_date, end_date):
    for n in range(int ((end_date - start_date).days)):
        yield start_date + timedelta(n)


def __generate_connections_between_integer(start, end, max_sentence_length=30):
    rest = end - start
    while True:
        if rest % max_sentence_length > 0:
            pass


def __generate_connections_between_integer(start, end, max_sentence_length=30):
    mylist = []
    for i in range(start, end):
        mylist.append(i)
        if len(mylist) > max_sentence_length:
        #yield str(i) + 'nextTo' str(i+1)
            pass



def generate_datatype_property_walks(literal_triples, binning=False):
    range_values = defaultdict(set)
    for s,p,o in literal_triples:
        range_values[p].add(o)
        yield ' '.join([s,p,o])



