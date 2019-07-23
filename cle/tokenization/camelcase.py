import regex

camel_case_pattern = regex.compile("(?V1)(?<=[a-z])(?=[A-Z])|(?<=[.!?]) +(?=[A-Z])")

string = "I have 9 sheep in my garageVideo games are super cool. Some peanuts can sing, though they taste a whole lot better than they sound!"
result = regex.split("(?V1)(?<=[a-z])(?=[A-Z])|(?<=[.!?]) +(?=[A-Z])", string)

def camel_case_split(text):
    return ' '.join(camel_case_pattern.split(text)).replace('_', ' ')
    #return ' '.join(camel_case_pattern.findall(text))#.replace('_', ' _ ')


if __name__ == '__main__':
    print(camel_case_split('ThisIsOneWord Am Anfang noch Grossbuchstaben. DiesIstEinTest diesIstEinTest und Weiter gehts. UNd weiter'))