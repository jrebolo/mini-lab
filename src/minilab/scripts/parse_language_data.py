
if __name__ == '__main__':
    """Parse raw data into english and japanese sentences and save them in separate files"""

    with open('/Users/joaovieira/fun/miny-lab/src/data/en_to_jp/raw', 'r', encoding="utf-8") as f:
        lines = f.readlines()
    words = []
    for line in lines:
        en, jp = line.strip().split('\t')
        words.append(en)
        words.append(jp)

    english_sentences = [word.strip() for word in words[0::2]]
    japanese_sentences = [word.strip() for word in words[1::2]]

    with open('../data/en_to_jp/english_sentences.txt', 'w', encoding="utf-8") as f:
        f.write('\n'.join(english_sentences))
    with open('../data/en_to_jp/japanese_sentences.txt', 'w', encoding="utf-8") as f:
        f.write('\n'.join(japanese_sentences))