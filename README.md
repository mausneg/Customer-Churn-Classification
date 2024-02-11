# Laporan Proyek Machine Learning - Maulana Surya Negara

## Domain Proyek

Churn pelanggan merujuk pada fenomena di mana pelanggan menghentikan hubungan atau langganan mereka dengan perusahaan atau penyedia layanan. Tingkat churn pelanggan berdampak langsung pada pendapatan, pertumbuhan, dan retensi pelanggan perusahaan. Oleh karena itu, memahami dan mengurangi churn pelanggan adalah prioritas utama bagi banyak bisnis

Misalnya, dalam industri telekomunikasi, pelanggan mungkin memilih untuk berhenti berlangganan jika mereka merasa tidak puas dengan kualitas layanan, menemukan penawaran yang lebih baik dari pesaing, atau merasa bahwa biaya langganan mereka tidak sebanding dengan nilai yang mereka terima. Dalam kasus seperti ini, perusahaan dapat kehilangan pendapatan signifikan dan mungkin harus mengeluarkan biaya tambahan untuk memperoleh pelanggan baru.

Dengan melakukan analisis prediksi churn, perusahaan dapat mengidentifikasi pelanggan yang berisiko churn dan mengambil tindakan proaktif untuk mempertahankan mereka. Misalnya, mereka mungkin menawarkan diskon atau peningkatan layanan untuk meningkatkan kepuasan pelanggan dan mencegah mereka berhenti berlangganan. Oleh karena itu, analisis prediksi churn adalah alat yang sangat berharga untuk mempertahankan pelanggan dan meningkatkan kinerja bisnis [^1^].

[^1^]: [Ahmadi, T., Wulandari, A., & Suhatman, H. Sistem Customer Churn Prediction Menggunakan Machine Learning pada Perusahaan ISP.](https://repository.pnj.ac.id/id/eprint/14345/3/JURNAL.pdf)

## Business Understanding

### Problem Statements

- Perusahaan mengalami tingkat churn pelanggan yang tinggi, yang berdampak negatif pada pendapatan dan pertumbuhan mereka.
- Perusahaan saat ini tidak memiliki cara untuk mengidentifikasi pelanggan yang berisiko churn.
- Tanpa kemampuan untuk memprediksi churn, perusahaan tidak dapat mengambil tindakan proaktif untuk mempertahankan pelanggan.

### Goals

- Membangun model prediktif yang dapat memprediksi churn pelanggan dengan akurasi di atas 85%.
- Mengidentifikasi fitur-fitur yang paling berkontribusi terhadap churn pelanggan.
- Menggunakan model untuk mengidentifikasi pelanggan yang berisiko churn.
- Mengambil tindakan proaktif berdasarkan prediksi model untuk mempertahankan pelanggan dan mengurangi churn.
- Meningkatkan retensi pelanggan dan, pada akhirnya, pendapatan dan pertumbuhan perusahaan.

## Data Understanding

Data yang digunakan pada proyek kali ini adalah data yang diambil dari <a href='https://www.kaggle.com/datasets/muhammadshahidazeem/customer-churn-dataset/data' target='_blank'>Kaggle</a>. Trainning file untuk dataset CHURN berisi kumpulan 505207 record pelanggan beserta fitur dan label churn mereka masing-masing. File ini berfungsi sebagai sumber utama untuk melatih model machine learning untuk memprediksi churn pelanggan. Setiap record dalam file training mewakili pelanggan dan mencakup fitur seperti CustomerID, Age Gender, Tenure, Usage Frequency, Support Calls, Payment Delay, Subscription Type, Contract Length, Total Spend, dan Last Interaction. Label churn menunjukkan apakah pelanggan telah churn (1) atau tidak (0). Dengan memanfaatkan file pelatihan ini, bisnis dapat mengembangkan model prediksi churn yang akurat untuk mengidentifikasi pelanggan yang paling mungkin untuk churn dan mengambil tindakan proaktif untuk mempertahankannya.

### Deskripsi Data

![alt text](images/image-2.png)

Berdasarkan informasi dari Kaggle, variabel-variabel pada Diamond dataset adalah sebagai berikut:

- `CustomerID`: ID unik untuk setiap pelanggan.
- `Age`: Usia pelanggan dengan rentang 18-65 tahun.
- `Gender`: Jenis kelamin pelanggan (Pria/Wanita).
- `Tenure`: Masa aktif pelanggan dengan rentang 1-60 bulan.
- `Usage Frequency`: Frekuensi penggunaan layanan dengan rentang 1-30 kali.
- `Support Calls`: Jumlah panggilan dukungan pelanggan dengan rentang 0-10 panggilan.
- `Payment Delay`: Jumlah hari keterlambatan pembayaran dengan rentang 0-30 hari.
- `Subscription Type`: Tipe langganan pelanggan seperti standar, basic, dan premium.
- `Contract Length`: Tipe kontrak pelanggan seperti bulanan, tahunan, dan 3 tahunan.
- `Total Spend`: Total pengeluaran pelanggan dalam dolar.
- `Last Interaction`: Waktu interaksi terakhir dengan pelanggan dalam hari.
- `Churn`: Label churn pelanggan (1: churn, 0: tidak churn).

Dari `info()` didapatkan informasi bahwa:

- Terdapat 3 kolom bertipe data `object` yaitu `Contract Length`, `Gender`, dan `Subscription Type`.
- Terdapat 8 kolom bertipe data `float` yaitu `CustomerID`, `Age`, `Tenure`, `Usage Frequency`, `Support Calls`, `Payment Delay`, `Total Spend`, `Last Interaction`, dan `Churn`.
- Terdapat 2 variabel yang bertipe data tidak sesuai, yaitu `Churn` dan `CustomerID`. Variabel `Churn` seharusnya bertipe data boolean atau dapat juga integer karena true/false akan diwakilkan oleh 1/0, sedangkan variabel `CustomerID` sedangkan variabel `CustomerID` seharusnya bertipe data string. Akan tetapi, variabel `CustomerID` tidak akan digunakan dalam proses analisis data, sehingga tidak perlu diubah tipe datanya.

![alt text](images/image-1.png)

Dari `describe()` dapat diketahui informasi sebagai berikut:

- Count: Jumlah data pada setiap kolom.
- Mean: Rata-rata dari setiap kolom.
- Std: Standar deviasi dari setiap kolom.
- Min: Nilai minimum dari setiap kolom.
- 25%: Kuartil pertama dari setiap kolom.
- 50%: Median dari setiap kolom.
- 75%: Kuartil ketiga dari setiap kolom.
- Max: Nilai maksimum dari setiap kolom.

Dengan min, max, median, dan mean dari setiap kolom dapat diketahui tidak ada nilai yang jauh dari nilai rata-rata, sehingga dapat disimpulkan bahwa tidak ada outlier pada dataset.

### Univariate Analysis

![alt text](images/image.png)

Dari hasil analisis univariate pada categorical features di atas, dapat diamati bahwa:

- Pada countplot dari variabel `Gender`, terlihat bahwa jumlah pelanggan laki-laki lebih banyak dibandingkan dengan pelanggan perempuan.
- Pada countplot dari variabel `Subscription Type`, terlihat bahwa sebaran pelanggan relatif merata pada masing-masing tipe subscription.
- Pada countplot dari variabel `Contract Length`, terlihat bahwa pelanggan lebih banyak yang memiliki kontrak tahunan dan triwulanan dibandingkan dengan kontrak bulanan.

![alt text](images/image-3.png)

Dari hasil analisis univariate pada numerical features di atas, dapat diamati bahwa:

- Pada histogram dari variabel `Age`, terlihat bahwa sebagian besar pelanggan berusia antara 20-50 tahun.
- Pada histogram dari variabel `Tenure` dan `Usage Frequency`, terlihat bahwa sebaran pelanggan relatif merata.
- Pada histogram dari variabel `Support Calls` terlihat bahwa sebagian besar pelanggan melakukan panggilan dukungan kurang dari 5 kali.
- Pada histogram dari variabel `Payment Delay`, terlihat bahwa sebagian besar pelanggan melakukan keterlambatan pembayaran kurang dari 20 hari.
- Pada histogram dari variabel `Total Spend`, terlihat bahwa sebagian besar pelanggan mengeluarkan biaya lebih dari 500 dolar.
- Pada histogram dari variabel `Last Interaction`, terlihat bahwa sebaran pelanggan relatif merata.

### Multivariate Analysis

Pada analisis multivariate, akan dilakukan analisis korelasi antar variabel pada dataset. Hal ini dilakukan untuk mengetahui korelasi antar variabel pada dataset sehingga dapat diketahui variabel mana saja yang memiliki korelasi tinggi dengan label churn.

![alt text](images/image-4.png)

Dari diagram heatmap korelasi di atas, dapat diamati bahwa:

- Hanya variabel `Support Calls` yang memiliki korelasi moderat dengan label churn.
- Variabel `Age`, `Payment Delay`, `Last Interaction`,dan `Total Spend` memiliki korelasi rendah dengan label churn.
- Sedangkan variabel `Tenure` dan `Usage Frequency` memiliki korelasi mendekati nol dengan label churn.

Pada kasus kali ini, karena model yang akan digunakan adalah deep learning, maka tidak variabel yang memiliki korelasi rendah dengan label churn tidak dihapus. Hal ini karena deep learning dapat memberikan weight yang tepat untuk setiap variabel sehingga tidak perlu melakukan feature selection.

## Data Preparation

Pada tahap ini, perlu dilakukan beberapa proses untuk mempersiapkan data sebelum dilakukan pemodelan. Proses-proses tersebut antara lain:

### Menghapus Kolom yang Tidak Diperlukan

Pada tahap ini, kolom `CustomerID` dihapus karena tidak diperlukan dalam proses pemodelan.

### Encoding Variabel Kategorikal

Pada fitur `Gender` female diubah menjadi 0 dan male diubah menjadi 1, sedangkan pada fitur `Subscription Type` dan `Contract Length` dilakukan one-hot encoding.

### Splitting Data

Dataset akan dibagi menjadi data training, data testing, dan data validation. Dengan ukuran data training sebesar 80%, data testing sebesar 10%, dan data validation sebesar 10%. Masing-masing ukuran ini dipertimbangkan dengan ukuran dataset yang ada. Sehingga didapatkan hasil bahwa data training sebesar 404164, data testing sebesar 50521, dan data validation sebesar 50521. Dengan ukuran 50521 pada data testing dan data validation, merupakan ukuran yang cukup besar untuk melakukan evaluasi model.

## Modeling

Pada proyek ini, akan digunakan model machine learning tensorflow dengan arsitektur deep learning. Hal ini karena beberapa alasan, antara lain:

- Kemampuan untuk menangkap pola yang kompleks: Deep learning mampu mempelajari pola yang sangat kompleks dalam data. Ini berarti bahwa jika ada hubungan non-linear atau interaksi antara fitur dalam dataset, model deep learning mungkin mampu menangkapnya.
- Kemampuan untuk menangani data berdimensi tinggi: Karena pada kasus ini memiliki sekitar 14 fitur sehingga model deep learning dapat mengelola kompleksitas ini.
- Pemilihan fitur otomatis: Deep learning dapat mempelajari fitur-fitur penting secara otomatis selama proses pelatihan karena pada kasus ini banyak fitur yang tidak memiliki korelasi yang tinggi dengan label churn.
- Performa: Dalam banyak kasus, deep learning telah menunjukkan performa yang sangat baik dalam berbagai tugas prediksi, sering kali melebihi model lain seperti regresi logistik atau random forest [^2^].
  [^2^]: [Sofia, R. N., & Supriyadi, D. (2021). Komparasi Metode Machine Learning dan Deep Learning untuk Deteksi Emosi pada Text di Sosial Media. JUPITER (Jurnal Penelitian Ilmu dan Teknik Komputer), 13(2), 130-139.](https://jurnal.polsri.ac.id/index.php/jupiter/article/view/3603/1677)

Pada proses pelatihan model akan dilakukan hyperparameter tuning dengan Graduate Student Descent (GSD). Hal ini karena GSD merupakan metode yang paling umum digunakan untuk melakukan hyperparameter tuning pada deep learning. Berikut merupakan hyperparameter yang akan di tuning:
- Optimizer: Optimizer mengontrol bagaimana model diperbarui berdasarkan data yang dilihat dan fungsi loss-nya. Pada kasus ini, akan dicoba menggunakan optimizer adam dengan learning rate 0.001 (default), 0.0001, dan 0.01.
- Jumlah Neuron pada Hidden Layer: Jumlah neuron dalam hidden layer dapat mempengaruhi kapasitas model untuk mempelajari pola dalam data. Terlalu sedikit neuron dapat menyebabkan underfitting, sementara terlalu banyak neuron dapat menyebabkan overfitting. Pada kasus ini, akan dicoba menggunakan 32, 64, 128, 256, 512, dan 1024.
- Jumlah Hidden Layer: Jumlah hidden layer dalam model deep learning juga dapat mempengaruhi kapasitas model. Model dengan lebih banyak layer dapat mempelajari pola yang lebih kompleks, tetapi juga lebih berisiko overfitting dan membutuhkan lebih banyak data untuk pelatihan. Pada kasus ini, akan dicoba menggunakan 1 sampai 3 hidden layer.
- Epoch: Epoch adalah jumlah kali seluruh dataset melalui jaringan neural selama pelatihan. Pada kasus ini, akan dicoba menggunakan 10 dan 20 epoch saja.
- Batch Size: Batch size adalah jumlah sampel yang diproses sebelum model diperbarui. Batch size yang lebih kecil dapat menghasilkan pembaruan yang lebih sering, tetapi juga dapat menyebabkan noise dalam pembaruan tersebut. Pada kasus ini, akan dicoba menggunakan batch size 128, 256, 512, dan 1024.

Dari proses tuning maka didapatkan model yang terbaik terdiri dari 1024 neuron, layer kedua terdiri dari 512 neuron, dan layer terakhir terdiri dari 1 neuron. Model ini menggunakan fungsi aktivasi relu pada layer pertama dan kedua, karena relu merupakan fungsi aktivasi yang paling umum digunakan pada hidden layer. Sedangkan pada layer terakhir menggunakan fungsi aktivasi sigmoid, karena sigmoid merupakan fungsi aktivasi yang paling umum digunakan pada output layer untuk klasifikasi biner.

Selanjutnya, model akan di compile dengan optimizer adam dengan learning rate default (0.001), loss function binary crossentropy, dan metrics accuracy. Optimizer adam merupakan optimizer yang paling umum digunakan pada deep learning karena dapat melakukan penyesuaian learning rate secara otomatis. Loss function binary crossentropy merupakan loss function yang paling umum digunakan pada klasifikasi biner. Metrics accuracy digunakan untuk mengukur performa model.

## Evaluation

Model akan dievaluasi dengan menggunakan validation dataset. Hal ini dilakukan untuk mengukur performa model pada data yang belum pernah dilihat sebelumnya. Model akan dievaluasi dengan menggunakan metrics accuracy. Sehingga didapatkan hasil akhir dari model yang telah dibuat yaitu 0.9155 pada train accuracy, 0.9193 pada test accuracy, dan 0.9160 pada val accuracy. Hal ini menunjukkan bahwa model yang telah dibuat memiliki performa yang baik sehingga tidak terjadi overfitting maupun underfitting.
