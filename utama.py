# ========================================================================================================
# Rekomendasi Buku Bagus
# ========================================================================================================
import pandas as pd 
import numpy as np 

df = pd.read_csv(
    'books.csv', 
    delimiter = ';',
    usecols = ['book_id', 'authors', 'original_title', 'title']
)

# ------------------------------------------------------------
# Menggabungkan kolom authors dan original title sebagai nilai feature (base content) dari setiap buku yang akan dijadikan acuan
def mergeCol(df):
    return str(df['authors']) + ' ' + str(df['original_title'])
df['features'] = df.apply(mergeCol, axis='columns')
 
# ------------------------------------------------------------
# Count feature
from sklearn.feature_extraction.text import CountVectorizer
model = CountVectorizer(
    tokenizer = lambda i: i.split(' ')
)

matrixFeatures = model.fit_transform(df['features'])
Features = model.get_feature_names()
jmlFeatures = len(Features)
eventFeatures = matrixFeatures.toarray()

# Cosine Similarity
from sklearn.metrics.pairwise import cosine_similarity
score = cosine_similarity(matrixFeatures)

# ------------------------------------------------------------
# Data favorit kelima orang (diasumsi bahwa buku yang menjadi input hanya buku yang memiliki rating 4 atau 5 saja ----> agar rekomendasi lebih akurat!)
favAndi = ['The Hunger Games', 'Catching Fire', 'Mockingjay', 'The Hobbit or There and Back Again']
favBudi = ["Harry Potter and the Philosopher's Stone", 'Harry Potter and the Chamber of Secrets', 'Harry Potter and the Prisoner of Azkaban']
favCiko = ['Robots and Empire']
favDedi = ['Nine Parts of Desire: The Hidden World of Islamic Women', 'A History of God: The 4,000-Year Quest of Judaism, Christianity, and Islam', 'No god but God: The Origins, Evolution, and Future of Islam']
favEllo = ['Doctor Sleep', 'The Story of Doctor Dolittle', "Bridget Jones's Diary (Bridget Jones, #1)"]      

listAndi = []
listBudi = []
listCiko = []
listDedi = []
listEllo = []
for i in favAndi:
    listAndi.append(df[df['original_title'] == i]['book_id'].tolist()[0]-1)
for i in favBudi:
    listBudi.append(df[df['original_title'] == i]['book_id'].tolist()[0]-1)
for i in favCiko:
    listCiko.append(df[df['original_title'] == i]['book_id'].tolist()[0]-1)
for i in favDedi:
    listDedi.append(df[df['original_title'] == i]['book_id'].tolist()[0]-1)
for i in favEllo:
    if i == "Bridget Jones's Diary (Bridget Jones, #1)":
        listEllo.append(df[df['title'] == i]['book_id'].tolist()[0]-1)
    else: 
        listEllo.append(df[df['original_title'] == i]['book_id'].tolist()[0]-1)

# ------------------------------------------------------------
# Mencari nilai score cosine similarity setiap buku terhadap masing-masing judul buku yang disukai setiap orang
daftarScoreAndi = []
for i in range(len(listAndi)):
    daftarScoreAndi.append(list(enumerate(score[listAndi[i]])))

daftarScoreBudi = []
for i in range(len(listBudi)):
    daftarScoreBudi.append(list(enumerate(score[listBudi[i]])))

daftarScoreCiko = []
for i in range(len(listCiko)):
    daftarScoreCiko.append(list(enumerate(score[listCiko[i]])))

daftarScoreDedi = []
for i in range(len(listDedi)):
    daftarScoreDedi.append(list(enumerate(score[listDedi[i]])))

daftarScoreEllo = []
for i in range(len(listEllo)):
    daftarScoreEllo.append(list(enumerate(score[listEllo[i]])))

# Mencari nilai akhir cosine similarity terhadap setiap buku yang ada untuk setiap orang (dengan cara merata-ratakan nilai cosine similarity dari setiap buku yang disukai setiap orang, yang sudah dicari sebelumnya)
listScoreAndi = []
for i in daftarScoreAndi[0]:
    listScoreAndi.append((
        i[0],
        (daftarScoreAndi[0][i[0]][1] + daftarScoreAndi[1][i[0]][1] + daftarScoreAndi[2][i[0]][1] + daftarScoreAndi[3][i[0]][1])/len(daftarScoreAndi)
    ))

listScoreBudi = []
for i in daftarScoreBudi[0]:
    listScoreBudi.append((
        i[0],
        (daftarScoreBudi[0][i[0]][1] + daftarScoreBudi[1][i[0]][1] + daftarScoreBudi[2][i[0]][1])/len(daftarScoreBudi)
    ))

listScoreCiko = []
for i in daftarScoreCiko[0]:
    listScoreCiko.append((
        i[0],
        (daftarScoreCiko[0][i[0]][1])/len(daftarScoreCiko)
    ))

listScoreDedi = []
for i in daftarScoreDedi[0]:
    listScoreDedi.append((
        i[0],
        (daftarScoreDedi[0][i[0]][1] + daftarScoreDedi[1][i[0]][1] + daftarScoreDedi[2][i[0]][1])/len(daftarScoreDedi)
    ))

listScoreEllo = []
for i in daftarScoreEllo[0]:
    listScoreEllo.append((
        i[0],
        (daftarScoreEllo[0][i[0]][1] + daftarScoreEllo[1][i[0]][1] + daftarScoreEllo[2][i[0]][1])/len(daftarScoreEllo)
    ))

# ------------------------------------------------------------
# Menyortir urutan buku berdasarkan nilai cosine similaritynya untuk direkomendasikan
bukuRekomenAndi = sorted(
    listScoreAndi,
    key = lambda x: x[1],
    reverse=True
)

bukuRekomenBudi = sorted(
    listScoreBudi,
    key = lambda x: x[1],
    reverse=True
)

bukuRekomenCiko = sorted(
    listScoreCiko,
    key = lambda x: x[1],
    reverse=True
)

bukuRekomenDedi = sorted(
    listScoreDedi,
    key = lambda x: x[1],
    reverse=True
)

bukuRekomenEllo = sorted(
    listScoreEllo,
    key = lambda x: x[1],
    reverse=True
)

# ------------------------------------------------------------
# Output (hanya menampilkan 5 buku terbaik dengan nilai cosine similaritynya > 0.3)
print('1. Buku bagus untuk Andi:')
for i in range(5):
    if bukuRekomenAndi[i][0] not in listAndi and bukuRekomenAndi[i][1] > 0.3:
        print('-', df['original_title'].iloc[bukuRekomenAndi[i][0]])
    else:
        i += 5
        print('-', df['original_title'].iloc[bukuRekomenAndi[i][0]])
print(' ')
print('2. Buku bagus untuk Budi:')
for i in range(5):
    if bukuRekomenBudi[i][0] not in listBudi and bukuRekomenBudi[i][1] > 0.3:
        print('-', df['original_title'].iloc[bukuRekomenBudi[i][0]])
    else:
        i += 5
        print('-', df['original_title'].iloc[bukuRekomenBudi[i][0]])
print(' ')
print('3. Buku bagus untuk Ciko:')
for i in range(5):
    if bukuRekomenCiko[i][0] not in listCiko and bukuRekomenCiko[i][1] > 0.3:
        if str(df['original_title'].iloc[bukuRekomenCiko[i][0]]) == 'nan':
            print('-', df['title'].iloc[bukuRekomenCiko[i][0]])
        else:
            print('-', df['original_title'].iloc[bukuRekomenCiko[i][0]])  
    else:
        i += 5
        if str(df['original_title'].iloc[bukuRekomenCiko[i][0]]) == 'nan':
            print('-', df['title'].iloc[bukuRekomenCiko[i][0]])
        else:
            print('-', df['original_title'].iloc[bukuRekomenCiko[i][0]])  
print(' ')
print('4. Buku bagus untuk Dedi:')
for i in range(5):
    if bukuRekomenDedi[i][0] not in listDedi and bukuRekomenDedi[i][1] > 0.3:
        print('-', df['original_title'].iloc[bukuRekomenDedi[i][0]])
    else:
        i += 5
        print('-', df['original_title'].iloc[bukuRekomenDedi[i][0]])

print(' ')
print('5. Buku bagus untuk Ello:')
for i in range(5):
    if bukuRekomenEllo[i][0] not in listEllo and bukuRekomenEllo[i][1] > 0.3:
        if str(df['original_title'].iloc[bukuRekomenEllo[i][0]]) == 'nan':
            print('-', df['title'].iloc[bukuRekomenEllo[i][0]])
        else:
            print('-', df['original_title'].iloc[bukuRekomenEllo[i][0]])  
    else:
        i += 5
        if str(df['original_title'].iloc[bukuRekomenEllo[i][0]] )== 'nan':
            print('-', df['title'].iloc[bukuRekomenEllo[i][0]])
        else:
            print('-', df['original_title'].iloc[bukuRekomenEllo[i][0]])  

