

def save_sentences(generator_func, file_name):
    with open(file_name, 'w', encoding='utf-8') as f:
        for sentence in generator_func:
            f.write(' '.join(sentence) + '\n')