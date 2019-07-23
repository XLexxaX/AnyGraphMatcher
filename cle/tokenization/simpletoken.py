from tokenization.camelcase import camel_case_split


def tokenize(text):
    whitespace_separated_text = camel_case_split(text.replace(',', ' ').replace(';', ' ').replace(':', ' ').replace('(', ' ').replace(')', ' ')
                     .replace('?', ' ').replace('!', ' ').replace('.', ' ').replace('_', ' ').replace('-', ' ').replace('&', ' '))
    return [word.lower() for word in whitespace_separated_text.split()]