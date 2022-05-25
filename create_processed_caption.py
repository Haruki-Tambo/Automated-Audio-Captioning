import nltk
import csv
from collections import defaultdict

# ステミングをする
def cap_stem(file_path):
    data = {}
    with open(file_path) as f:
        reader = csv.reader(f)
        for row in reader:
            data[row[0]] = row[1:]
        data.pop('file_name')

    stemmer = nltk.stem.SnowballStemmer('english')
    for audio_file_name, cap in data.items():
        for i in range(len(cap)):
            words = cap[i].split()
            for j in range(len(words)):
                words[j] = words[j].replace(',', '')
                words[j] = words[j].replace('.', '')
                words[j] = stemmer.stem(words[j])
            cap[i] = ' '.join(words)

    return data

#単語を数える
def word_cnt(stemed_cap):
    rank = defaultdict(int)
    for audio_file_name, cap in stemed_cap.items():
        for i in range(len(cap)):
            words = cap[i].split()
            for j in range(len(words)):
                rank[words[j]] += 1
            cap[i] = ' '.join(words)
    return rank

# ２文字以下の単語と上位２０の単語と上位３００の単語以外をrankから削除する
def exclud_words(rank):
    new_rank = []
    rank_list = sorted(rank.items(), key=lambda x:x[1], reverse=True)
    for i in range(20, len(rank_list)):
        if len(rank_list[i][0]) <= 2:
            continue
        else:
            new_rank.append(rank_list[i][0])
        if len(new_rank) >= 300:
            break

    return new_rank

# キャプションを１つにまとめて上位３００の単語以外をキャプションから削除する
def exclud_cap(rank, stemed_cap):
    rank = set(rank)
    for audio_file_name, cap in stemed_cap.items():
        cap = set(' '.join(cap).split())
        cap = list(cap & rank)
        stemed_cap[audio_file_name] = cap
    return stemed_cap

# csvファイルに出力する
def output_csv(processed_data, name):
    with open(f'data/{name}.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['file_name', 'words'])
        for audio_file_name, words in processed_data.items():
            writer.writerow([audio_file_name, ' '.join(words)])

# word2vecを学習させるためのテキストデータを作る
def create_target_development(file_path, word_num):
    data = {}
    with open(file_path) as f:
        reader = csv.reader(f)
        for row in reader:
            # data[row[0]] = [txt.lower().replace(',', '').replace('.', '') for txt in row[1:]]
            for i in range(1, len(row)):
                row[i] = row[i].lower().replace(',', '').replace('.', '')
                while len(row[i].split()) < word_num:
                    row[i] += ' <pad>'
            data[row[0]] = row[1:]

        data.pop('file_name')

    with open('data/target_development_pad.txt', 'w') as f:
        first = 1
        for k, captions in data.items():
            if first:
                captions = ('\n').join(captions)
                first = 0
            else:
                captions = '\n'+('\n').join(captions)
            f.write(captions)

file_path_dev = 'data/clotho_captions_development.csv'
file_path_eva = 'data/clotho_captions_evaluation.csv'

# development
# stemed_dev = cap_stem(file_path_dev)
# rank = word_cnt(stemed_dev)
# top_300_words = exclud_words(rank)
# processed_dev = exclud_cap(top_300_words, stemed_dev)
# output_csv(processed_dev, 'processed_captions_development')

# evaluation 上位３００の単語の処理は必要なし
# stemed_eva = cap_stem(file_path_eva)
# rank = word_cnt(stemed_eva)
# processed_eva = exclud_cap(top_300_words, stemed_eva)
# output_csv(processed_eva, 'processed_captions_evaluation')

# word2vecを学習させるためのテキストデータ
# linesentenceに入力する
WORD_NUM = 20
create_target_development(file_path_dev, WORD_NUM)