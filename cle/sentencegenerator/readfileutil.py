

def read_from_file_generator(sentences_filename):
    for line in open(sentences_filename, "r"):
        yield(line)

def read_from_file(sentences_filename, properties):
    sentences = list()
    for line in open(sentences_filename, "r", encoding="UTF-8"):
        line = line.replace('"',"").replace(" .\n","").replace("<","").replace(">","")
        line = line.lower().split(" ")
        if properties is not None:
            l = list()
            has_to_be_in_properties = False
            for word in line:
                if word in properties:
                    l.append(word)
                elif not has_to_be_in_properties:
                    has_to_be_in_properties = True
                    l.append(word)
                else:
                    break
            if len(l) > 1:
                sentences.append(l)
        else:
            sentences.append(line)
    return sentences
