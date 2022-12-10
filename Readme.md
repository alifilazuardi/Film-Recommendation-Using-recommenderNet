# Laporan Proyek  Akhir Machine Learning Terapan - Alifi Lazuardi Gunawan

## Domain Proyek

Sistem rekomendasi adalah suatu mekanisme yang bekerja dengan tujuan untuk mempermudah pengguna dalam mendapatkan atau memilih produk sesuai dengan keiinginan. Sistem rekomendasi bekerja dengan memberikan beberapa saran pilihan produk kepada pengguna berdasarkan algoritma tertentu. Sistem rekomendasi dapat dibangun berdasarkan beberapa metode seperti *collaborative filtering, content based filtering, knowledge based filtering, dan hybrid filtering*. 

Dalam proyek ini akan dikembangkan sistem rekomendasi yang menggunakan metode *content based filtering* dan *collaborative filtering*. Metode  *content based filtering* menggunakan kemiripan dari sebuah fitur yang ada dalam suatu hal yang akan direkomendasikan (Girsang, 2020). Dalam Proyek ini sistem rekomendasi yang dibagun berdasarkan pada genre dari film. *Collaborative filtering* akan merekomendasikan konten kepada target berdasarkan ketertarikan orang lain, jadi sistem akan mencari orang dengan ketertarikan sama kemudian merekomendasikan apa yang disukai orang tersebut kepada target (Handrico, 2012). 

Berdasarkan pemaparan di atas proyek ini akan melakukan pembuatan model *content based recommendation system* dan *collaborative filtering recommnedation* pada dataset *MovieLens 20M Dataset* .

## Business Understanding

### a. *Problem Statement*

1. Bagaimana tahapan membangun sistem rekomendasi dengan metode *content based filtering* pada *MovieLens 20M Dataset* ?
1. Bagaimana tahapan membangun sistem rekomendasi dengan metode *collaborative filtering* pada *MovieLens 20M Dataset* ?
1. Apa hasil dari sistem rekomendasi dengan metode *content based filtering* dan *collaborative filtering*?

### b. *Goals*

1. Mengetahui tahapan dalam membangun sistem rekomendasi dengan metode *content based filtering* pada *MovieLens 20M Dataset*.
1. Mengetahui tahapan dalam membangun sistem rekomendasi dengan metode *collaborative filtering* pada *Comics MovieLens 20M Dataset*
1. Mendapatkan hasil rekomendasi film dengan metode *content based filtering* dan *collaborative filtering*.

### c. Pernyataan solusi

1. Untuk memperisapkan data beberapa tahapan yaitu:

   a. Membaca data movie dan rating dengan pandas dan meyimpannya dalam variabel *movies* dan *ratings*

   b. Mengambil sample data rating sebanyak 500000 data

   ​	Hal ini dilakukan karena jumlah data rating terlalu banyak. Sebelum dilakukna sampling data dilakukan pengacakan data dengan fungsi sample(random_state 42). Selanjutnya diambil 500000 data pertama dengan menggunakan fungsi head (500000).

   c. Menggabungkan data *movies* dan *ratings*

   ​	pennggabungan data dilakukan dengan menggunakan fungsi pandas.merge. Penggabungan didasarkan pada kolom movieId.

   d. Melakukan *data preparation* untuk model *content based filtering*

   e. melakukan *data preparation* untuk model *collaborative filtering*

2. Membangun sistem rekomendasi dengan metode *content based filering* berdasarkan fitur genre film dengan menggunakan algoritma *cosine similarity*.

3. Membangun sistem rekomendasi dengan metode *collaborative filering* berdasarkan rating pengguna lain dengan menggunakan *RecommenderNet* dari *TensorFlow*.

## Data Understanding

Data yang digunakan dalam proyek ini  berasal dari Kaggel dengan judul MovieLens 20M Dataset*. Dataset ini merupakan data  yang berisi daftar film beserta genre, rating dari user, dan tag. Dalam dataset ini terdapat beberapa file yaitu:

1. tag.csv yang berisi tag dari film yang diberikan oleh user
2. rating.csv yang berisi rating dari film yang diberikan oleh user
3. movie.csv yang berisi informasi dari film
4. link.csv yang berisi identifier untuk sumber lain seperti iMDB
5. genome_score.csv yang berisi relevansi tag dari film
6. genome_tag.csv yang berisi deskripsi dari tag

Dalam proyek ini data yang digunakan adalah rating.csv dan movie.csv. Dataset ini dapat dilihat pada link berikut [MovieLens 20M Dataset | Kaggle](https://www.kaggle.com/datasets/grouplens/movielens-20m-dataset?select=rating.csv)

### a. Variabel-Variabel Dari  Dataset

Variabel-varabel yang ada dalam data movie adalah: 

* *movieId* : Id dari film
* *title* : judul film
* *genres* : genre film

Variabel-varabel yang ada dalam data rating adalah: 

* uuserId : Id pengguna pemberi nilai
* *movieId* : Id dari film
* *rating* : nilai dari pengguna
* *timestamp* : waktu pemberian nilai

### b. *Univariate Exploratory Data Analysis*

#### 1. Pengecekan Kondisi data

​	a. Mendapatkan data movie unik berdasarkan movieId dan user unik berdasarkan userId

```
27278
```

```
26744
```

#### 2. Pengecekan *missing value* dan data duplikat

​	a. Pengecekan missing value 

Untuk mengecek value difunakan fungsi isna(). Fungsi ini akan menegembalikan nilai apabila terdapat null value. Selanjutnya dilakukan penjumlahan dengan fungsi sum() untuk menjumlahkan nilai null pada tiap kolom/fitur.

Jumlah *missing value* pada data *movie*

```
movieId    0
title      0
genres     0
dtype: int64
```

Jumlah *missing value*  pada data *rating*

```
userId       0
movieId      0
rating       0
timestamp    0
dtype: int64
```

​		b. Pengecekan data duplikat

Pengecekan duplikat dilakukan dengan menggunakan fungsi duplicated(). Fungsi ini akan mengembalikan nilai apabila terdapat nilai duplikat. Selanjutnya dilakukan penjumlahan dengan menggunakan fungsi sum() untuk mendapatkan total dari nilai duplikat.

```
0
```

​			Tidak terdapat data duplikat pada kedua data.

## Data Preparation

### a. *Data preparation* untuk *content based filtering*
1. Mengambil data pada kolom *movieId, title,* dan *genres*

   Model *content based filtering *yang akan dibangun akan berdaasrakan pada genre dari film, karena itu perlu dilakukan pengambilan data dan menyimpannya ke dalam variabel data_cb

   ```
   movieId	title	genres
   0	1	Toy Story (1995)	Adventure|Animation|Children|Comedy|Fantasy
   1	1	Toy Story (1995)	Adventure|Animation|Children|Comedy|Fantasy
   2	1	Toy Story (1995)	Adventure|Animation|Children|Comedy|Fantasy
   3	1	Toy Story (1995)	Adventure|Animation|Children|Comedy|Fantasy
   4	1	Toy Story (1995)	Adventure|Animation|Children|Comedy|Fantasy
   ...	...	...	...
   499995	130340	Wolfsburg (2003)	Drama
   499996	130502	Venus & Fleur (2004)	(no genres listed)
   499997	130512	Hippocrates (2014)	Comedy|Drama
   ```

2. Mengecek data duplikat dan menghapusnya

   Data duplikat terjadi karena satu movieId diberi rating oleh lebih dari 1 userId. Karena kolom userId tidak digunakan maka data movieId dianggap data duplikat.

   ```
   486848
   ```

   Terdapat 486848 buah data duplikat. Data duplikat selanjutnya dihapus menggunakan perintah drop_duplicates()

   ```
   0
   ```

   Data duplikat telah dihapus

### b. *Data preparation* unutk *collaborative filtering*

1. Membuat data_user_rating yang berasal dari data yang telah dibuat sebelumnya
	
   ```
	movieId	title	genres	level_0	index	userId	rating	timestamp
	0	1	Toy Story (1995)	Adventure|Animation|Children|Comedy|Fantasy	298489	8676	80	3.0	1997-03-15 20:41:53
	1	1	Toy Story (1995)	Adventure|Animation|Children|Comedy|Fantasy	16786	36852	285	2.5	2014-02-16 01:36:11
	2	1	Toy Story (1995)	Adventure|Animation|Children|Comedy|Fantasy	59367	85133	604	3.5	2007-05-15 03:14:36
	3	1	Toy Story (1995)	Adventure|Animation|Children|Comedy|Fantasy	61170	97713	683	5.0	1996-05-14 15:23:39
	4	1	Toy Story (1995)	Adventure|Animation|Children|Comedy|Fantasy	274471	158501	1059	3.0	1996-06-17 09:43:53
	...	...	...	...	...	...	...	...	...
	499995	130340	Wolfsburg (2003)	Drama	484000	14917897	103068	3.0	2015-03-16 19:23:46
	499996	130502	Venus & Fleur (2004)	(no genres listed)	433131	5400044	37105	0.5	2015-03-21 03:56:34
	499997	130512	Hippocrates (2014)	Comedy|Drama	7513	1317741	8963	3.5	2015-03-28 03:24:48
	499998	130970	George Carlin: Life Is Worth Losing (2005)	Comedy	471670	12478140	86211	5.0	2015-03-27 21:09:26
	499999	131092	Mickey, Donald, Goofy: The Three Musketeers (2...	Adventure|Animation|Children|Comedy	183903	11528449	79570	3.0	2015-03-29 18:57:42
	```
	
2. Mengubah userId dan movieId menjadi list tanpa nilai yang sama

3. Mapping data userId dan movieId yang diencode ke dalam data data_user_rating

      ```
      movieId	title	genres	level_0	index	userId	rating	timestamp	user	movie
      0	1	Toy Story (1995)	Adventure|Animation|Children|Comedy|Fantasy	298489	8676	80	3.0	1997-03-15 20:41:53	0	0
      1	1	Toy Story (1995)	Adventure|Animation|Children|Comedy|Fantasy	16786	36852	285	2.5	2014-02-16 01:36:11	1	0
      2	1	Toy Story (1995)	Adventure|Animation|Children|Comedy|Fantasy	59367	85133	604	3.5	2007-05-15 03:14:36	2	0
      3	1	Toy Story (1995)	Adventure|Animation|Children|Comedy|Fantasy	61170	97713	683	5.0	1996-05-14 15:23:39	3	0
      ```


4. Mencari jumlah user dan movie

   ```
   106761   jumlah user
   13152    jumlah movie
   ```

5. Mencarai nilai maksimal dan minimal dari rating

   ```
   Min Rating: 0.5, Max Rating: 5.0
   ```

6. Mengacak data sebelum proses *splitting*

   ```
   movieId	title	genres	level_0	index	userId	rating	timestamp	user	movie
   104241	605	One Fine Day (1996)	Drama|Romance	365421	17657636	122134	2.0	2000-12-10 03:54:58	30021	586
   199676	1499	Anaconda (1997)	Action|Adventure|Thriller	273350	19138269	132403	2.0	2001-03-13 00:09:50	52574	1390
   ```

7. Membuat data latih dan data validasi dengan rasio 8:2

   ```
   [[30021   586]
    [52574  1390]
    [ 9313   996]
    ...
    [57062   891]
    [25587  1031]
    [ 1093   797]] [0.33333333 0.33333333 0.55555556 ... 0.77777778 0.66666667 0.55555556]
   ```

   

## Modeling

### a.  *Content based filtering model*

1. Perhitungan *TF-IDF*

   *Term frequency-inverse document freuency* adalah perhitungan kemunculan sebuah term terhadap inverse dari kemunculan sebuah dokumen. *TF-IDF* menggambarkan seberapa penting sebuah term yang muncul dalam suatu dokumen dalam kumpulan dokumen atau korpus. *TF-IDF* dilakukan terhadap kolom genres dengan menggunakan *TfidfVectorizer()* dari *skleran*. Keluaran dari fungsi ini berupa matriks yang menyimpan nilai *float* yang disimpan dalam variabel *tf_float*

   ```
   <13152x24 sparse matrix of type '<class 'numpy.float64'>'
   	with 29480 stored elements in Compressed Sparse Row format>
   ```

2. Perhitungan *Cosine Similarity*

   *Cosine similarity* adalah ukuran kesamaan suatu *term* terhadap *term* yang lain. *Cosine similarity* digunakan untuk mencari kemiripan *query* yang merupakan inputan dari genre film dibandingkan dengan kumpulan genre film yang lain. Nilai *cosine similarity* akan digunakan untuk menemukan rekomendasi terdekat. Perhitungan nilai *cosine similarity* dilakukan dengan menggunakan fungsi *cosine_similarity()* dari *sklearn*.

   ```
   array([[1.        , 0.80440038, 0.156367  , ..., 0.2028521 , 0.26243901,
           0.87884992],
          [0.80440038, 1.        , 0.        , ..., 0.        , 0.        ,
           0.59330761],
          [0.156367  , 0.        , 1.        , ..., 0.46054051, 0.59582225,
           0.1779223 ],
          ...,
          [0.2028521 , 0.        , 0.46054051, ..., 1.        , 0.77294951,
           0.23081541],
          [0.26243901, 0.        , 0.59582225, ..., 0.77294951, 1.        ,
           0.29861641],
          [0.87884992, 0.59330761, 0.1779223 , ..., 0.23081541, 0.29861641,
           1.        ]])
   ```

3. Fungsi rekomendasi

   Fugsi rekomendasi berguna untuk mendapatkan rekomendasi dari sebuah film berdasarkan pada perhitungan *cosine similarity*. Fungsi ini menerima input berupa:

   - query : judul dari film yang dicari rekomendasinya

   - data : data film

   - n_recom : jumlah rekomendasi yang diinginkan

   - cos_sim : perhitungan nilai *cosine similarity*

   Contoh keluaran fungsi rekomendasi dengan jumlah rekomendasi sebanyak 20 buah

   ```
   Mencari rekomendaasi dari  Secret, A (Un secret) (2007) dengan genre  Drama|War
   Judul	genre
   0	Born on the Fourth of July (1989)	Drama|War
   1	Battleship Potemkin (1925)	Drama|War
   2	Ay, Carmela! (¡Ay, Carmela!) (1990)	Drama|War
   3	Charge of the Light Brigade, The (1968)	Drama|War
   4	Earth (1998)	Drama|War
   5	Stop-Loss (2008)	Drama|War
   6	Good (2008)	Drama|War
   7	Tin Drum, The (Blechtrommel, Die) (1979)	Drama|War
   8	Ivan's Childhood (a.k.a. My Name is Ivan) (Iva...	Drama|War
   9	Ararat (2002)	Drama|War
   10	Rome, Open City (a.k.a. Open City) (Roma, citt...	Drama|War
   11	Stalingrad (1993)	Drama|War
   12	Land Girls, The (1998)	Drama|War
   13	Flags of Our Fathers (2006)	Drama|War
   14	Leopard, The (Gattopardo, Il) (1963)	Drama|War
   15	Richard III (1955)	Drama|War
   16	Damned, The (La Caduta degli dei) (1969)	Drama|War
   17	Eagle Has Landed, The (1976)	Drama|War
   18	Coming Home (1978)	Drama|War
   19	Back to Bataan (1945)	Drama|War
   ```

### b.  *Collaborative filtering model*

1. Model *RecommederNet tensorflow*

   *RecommenderNe*t bekerja dengan menghitung skor kecocokan antara movie dan user menggunaka perkalian *dot product* dan menggunakan *bias* pada perhitungannya . *RecommenderNet* menerima data masukan berupa data *user* dan *rating* film yang dilakukan embedding menjadi vektor. Nilai hasil perhitungan akan dimasukkan ke dalam fungsi sifmoid untuk mendaptkan rentang antara 0-1.  

2. *Compile* pada model

   Model di*-compile* dengan menggunakan *binary crossentropy* untuk perhitungan *loss*, *Adam* sebagai *optimizer*, dan *RootMeanSquaredError* sebagai matriks evaluasi.

3.  *Training* pada model

   Proses *training*  dilakukan sebanyak 5 kali epoch.

   ```
   Epoch 1/5
   50000/50000 [==============================] - 295s 6ms/step - loss: 0.6273 - root_mean_squared_error: 0.2277 - val_loss: 0.6124 - val_root_mean_squared_error: 0.2121
   Epoch 2/5
   50000/50000 [==============================] - 293s 6ms/step - loss: 0.6042 - root_mean_squared_error: 0.2034 - val_loss: 0.6077 - val_root_mean_squared_error: 0.2073
   Epoch 3/5
   50000/50000 [==============================] - 278s 6ms/step - loss: 0.5977 - root_mean_squared_error: 0.1966 - val_loss: 0.6059 - val_root_mean_squared_error: 0.2056
   Epoch 4/5
   50000/50000 [==============================] - 279s 6ms/step - loss: 0.5937 - root_mean_squared_error: 0.1923 - val_loss: 0.6053 - val_root_mean_squared_error: 0.2048
   Epoch 5/5
   50000/50000 [==============================] - 282s 6ms/step - loss: 0.5907 - root_mean_squared_error: 0.1892 - val_loss: 0.6050 - val_root_mean_squared_error: 0.2046
   ```

4. Fungsi rekomendasi

   Hasil rekomendasi

   ```
   370/370 [==============================] - 1s 3ms/step
   Showing recommendations for users: 64271
   ===========================
   movie with high ratings from user
   --------------------------------
   Monty Python and the Holy Grail (1975) : Adventure|Comedy|Fantasy
   Ferris Bueller's Day Off (1986) : Comedy
   Memento (2000) : Mystery|Thriller
   Children of Men (2006) : Action|Adventure|Drama|Sci-Fi|Thriller
   WALL·E (2008) : Adventure|Animation|Children|Romance|Sci-Fi
   --------------------------------
   Top 10 movie recommendation
   --------------------------------
   Persuasion (1995) : Drama|Romance
   Sunset Blvd. (a.k.a. Sunset Boulevard) (1950) : Drama|Film-Noir|Romance
   High Noon (1952) : Drama|Western
   Big Sleep, The (1946) : Crime|Film-Noir|Mystery
   Koyaanisqatsi (a.k.a. Koyaanisqatsi: Life Out of Balance) (1983) : Documentary
   My Life as a Dog (Mitt liv som hund) (1985) : Comedy|Drama
   Celebration, The (Festen) (1998) : Drama
   Grand Illusion (La grande illusion) (1937) : Drama|War
   Double Indemnity (1944) : Crime|Drama|Film-Noir
   Intouchables (2011) : Comedy|Drama
   ```

## Evaluation

### a.  Evaluasi *Content based filtering model*

Evaluasi dilakukan dengan menghitung akurasi, yaitu kesesuaian genre rekomendasi dari model dengan genre dari *query* yang diberikan:

P = t/n

P = nilai presisi

t = jumlah rekomendasi benar

n = jumlah keseluruhan rekomendasi

Model yang dibuat memberikan 20 rekomendasi terhadap *query* yang mempunyai genre Drama|War. Jumlah rekomendasi bergenre Drama|War adalah 20 rekomendasi.

P = 20/20

P = 1

Model mampu membuat rekomendasi dengan nilai akurasi 1. Nilai akurasi akan sangat berpengaruh dengan jumlah film yang mempunyai genre sama dengan *query* dan jumlah rekomendasi yang diinginkan. Apabila jumlah rekomendasi yang diinginkan melebihi jumlah film dengan genre yang sama nilai akurasi akan menurun.

### b. Evaluasi  *Collaborative filtering* 

Evaluasi dilakukan dengan menggunakan *Root Mean Squared Error (RMSE)*. Nilai RMSE didapatkan dari nilai eror dibagi jumlah data kemudian dikuadratkan. Perhitungan RMSE dapat diperoleh dari variabel *history* yang merupakan proses *training* dari model.

![training](https://user-images.githubusercontent.com/68947748/204186083-8a95916e-3f2b-4c74-b0af-e6bd95ef665c.png)

Nilai RMSE yang dihasilkan sudah cukup rendah. Hal ini berarti model yang dibuat sudah cukup baik. Namun, terlihat model mengalamai *overfit* dimana nilai *RMSE test* naik seiring dengan *epoch*, sedangkan nilai *RMSE train* mengalami penurunan. *Overfit* terjadi ketika model terlalu spesifik memahami data latih sehingga tidak mendapatkan pola data secara general dan melakukan kesalahan prediksi ketika menghadapi data yang berbeda.

## Kesimpulan

Untuk membangun sistem rekomendasi dengan metode content based filtering diperlukan data dengan fitur tertentu sebagai dasar pemberian rekomendasi. Fitur ini digunakan untuk mengukur seberapa dekat *query* dengan kumpulan data yang akan direkomendasikan. Perhitungan kedekatan ini dilakukan dengan menggunakan *TF-IDF* dan *cosine similarity*. TF-IDF menghitung nilai seberapa penting sebuah term yang muncul dalam suatu dokumen dalam kumpulan dokumen atau korpus. *Cosine similarity* digunakan untuk mencari kemiripan *query* yang merupakan inputan dari genre film dibandingkan dengan kumpulan genre film yang lain. Dalam pengujuan model mampu merekomendasikan film dengan genre yang sama dengan akurasi mencapai 100%.

*Collaborative filtering* menggunakan kemiripan preferensi konten pengguna lain sebagai rekomendasi. Sistem akan mencari orang dengan ketertarikan sama kemudian merekomendasikan apa yang disukai orang tersebut kepada target. Dalam model yang dibangun kemiripan preferensi dibangun berdasarkan nilai rating yang diberikan. Sistem dibangun menggunakan *RecommenderNet* dari *tensorflow*. Berdasarkan hasil pelatihan diperoleh nilai RMSE sebesar 0.1892 pada data latih dan 0.2046 pada data validasi. Hal ini menunjukkan model yang dibuat cukup baik.

## Daftar Referensi

Banerjee, S., 2020. *Collaborative Filtering for Movie Recommendations.* [Online] 
 Available at: https://keras.io/examples/structured_data/collaborative_filtering_movielens/
 [Accessed 24 Movember 2022].

Girsang, A. S., 2020. *Sistem rekomendasi- Content Based.* [Online] 
 Available at: https://mti.binus.ac.id/2020/11/17/sistem-rekomendasi-content-based/
 [Accessed 23 Movember 2022].

Handrico, A., 2012. *SISTEM REKOMENDASI BUKU PERPUSTAKAAN FAKULTAS SAINS DAN TEKNOLOGI DENGAN METODE COLLABORATIVE FILTERING.* Pekanbaru: Jurusan teknik informatika, Fakultas Sains dan Teknologi Universitas Islam Negeri Sultan Syarif Kasim .

 