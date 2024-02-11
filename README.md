# Laporan Proyek Machine Learning - Maulana Surya Negara

## Domain Proyek

_Customer churn_ merujuk pada fenomena di mana pelanggan menghentikan hubungan atau langganan mereka dengan perusahaan atau penyedia layanan. Tingkat _churn_ pelanggan berdampak langsung pada pendapatan, pertumbuhan, dan retensi pelanggan perusahaan. Oleh karena itu, memahami dan mengurangi _churn_ pelanggan adalah prioritas utama bagi banyak bisnis

Misalnya, dalam industri telekomunikasi, pelanggan mungkin memilih untuk berhenti berlangganan jika mereka merasa tidak puas dengan kualitas layanan, menemukan penawaran yang lebih baik dari pesaing, atau merasa bahwa biaya langganan mereka tidak sebanding dengan nilai yang mereka terima. Dalam kasus seperti ini, perusahaan dapat kehilangan pendapatan signifikan dan mungkin harus mengeluarkan biaya tambahan untuk memperoleh pelanggan baru.

Dengan melakukan analisis prediksi _churn_, perusahaan dapat mengidentifikasi pelanggan yang berisiko _churn_ dan mengambil tindakan proaktif untuk mempertahankan mereka. Misalnya, mereka mungkin menawarkan diskon atau peningkatan layanan untuk meningkatkan kepuasan pelanggan dan mencegah mereka berhenti berlangganan. Oleh karena itu, analisis prediksi _churn_ adalah alat yang sangat berharga untuk mempertahankan pelanggan dan meningkatkan kinerja bisnis [^1^].

## Business Understanding

### Problem Statements

- Perusahaan mengalami tingkat _churn_ pelanggan yang tinggi, yang berdampak negatif pada pendapatan dan pertumbuhan mereka.
- Perusahaan saat ini tidak memiliki cara untuk mengidentifikasi pelanggan yang berisiko _churn_.
- Tanpa kemampuan untuk memprediksi _churn_, perusahaan tidak dapat mengambil tindakan proaktif untuk mempertahankan pelanggan.

### Goals

- Membangun model prediktif yang dapat memprediksi _churn_ pelanggan dengan akurasi di atas 85%.
- Mengidentifikasi fitur-fitur yang paling berkontribusi terhadap _churn_ pelanggan.
- Menggunakan model untuk mengidentifikasi pelanggan yang berisiko _churn_.
- Mengambil tindakan proaktif berdasarkan prediksi model untuk mempertahankan pelanggan dan mengurangi _churn_.
- Meningkatkan retensi pelanggan dan, pada akhirnya, pendapatan dan pertumbuhan perusahaan.

## Data Understanding

Data yang digunakan pada proyek kali ini adalah data yang diambil dari <a href='https://www.kaggle.com/datasets/muhammadshahidazeem/customer-churn-dataset/data' target='_blank'>Kaggle</a>. _Trainning_ _file_ untuk _dataset churn_ berisi kumpulan 505207 _record_ pelanggan beserta fitur dan label _churn_ mereka masing-masing. _File_ ini berfungsi sebagai sumber utama untuk melatih model _machine learning_ untuk memprediksi _churn_ pelanggan. Setiap _record_ dalam _file_ _training_ mewakili pelanggan dan mencakup fitur seperti `CustomerID`, `Age`, `Gender`, `Tenure`, `Usage Frequency`, `Support Calls`, `Payment Delay`, `Subscription Type`, `Contract Length`, `Total Spend`, dan `Last Interaction`. Label _churn_ menunjukkan apakah pelanggan telah _churn_ (1) atau tidak (0). Dengan memanfaatkan _file_ pelatihan ini, bisnis dapat mengembangkan model prediksi _churn_ yang akurat untuk mengidentifikasi pelanggan yang paling mungkin untuk _churn_ dan mengambil tindakan proaktif untuk mempertahankannya.

### Deskripsi Data

Berdasarkan informasi dari Kaggle, variabel-variabel pada _customer churn dataset_ adalah sebagai berikut:

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
- `Churn`: Label _churn_ pelanggan (1: _churn_, 0: tidak _churn_).

![alt text](https://github.com/mausneg/Customer-Churn-Classification/blob/main/images/image-2.png)

Gambar 1. Deskripsi Data _Customer Churn_

Dari Gambar 1 didapatkan informasi bahwa:

- Terdapat 3 kolom bertipe data `object` yaitu `Contract Length`, `Gender`, dan `Subscription Type`.
- Terdapat 8 kolom bertipe data `float` yaitu `CustomerID`, `Age`, `Tenure`, `Usage Frequency`, `Support Calls`, `Payment Delay`, `Total Spend`, `Last Interaction`, dan `Churn`.
- Terdapat 2 variabel yang bertipe data tidak sesuai, yaitu `Churn` dan `CustomerID`. Variabel `Churn` seharusnya bertipe data _boolean_ atau dapat juga _integer_ karena _true_/_false_ akan diwakilkan oleh 1/0, sedangkan variabel `CustomerID` sedangkan variabel `CustomerID` seharusnya bertipe data string. Akan tetapi, variabel `CustomerID` tidak akan digunakan dalam proses analisis data, sehingga tidak perlu diubah tipe datanya.

![alt text](https://github.com/mausneg/Customer-Churn-Classification/blob/main/images/image-1.png)

Gambar 2. Statistik Deskriptif Customer _Churn_

Dari Gambar 2, dapat diketahui informasi sebagai berikut:

- `Count`: Jumlah data pada setiap kolom.
- `Mean`: Rata-rata dari setiap kolom.
- `Std`: Standar deviasi dari setiap kolom.
- `Min`: Nilai minimum dari setiap kolom.
- `25%`: Kuartil pertama dari setiap kolom.
- `50%`: Median dari setiap kolom.
- `75%`: Kuartil ketiga dari setiap kolom.
- `Max`: Nilai maksimum dari setiap kolom.

Dengan _min_, _max_, median, dan mean dari setiap kolom dapat diketahui tidak ada nilai yang jauh dari nilai rata-rata, sehingga dapat disimpulkan bahwa tidak ada _outlier_ pada dataset.

### Univariate Analysis

![alt text](https://github.com/mausneg/Customer-Churn-Classification/blob/main/images/image.png)

Gambar 3. Univariate Analysis _Customer Churn_

Dari Gambar 3 hasil analisis _univariate_ pada _categorical features_ di atas, dapat diamati bahwa:

- Pada _countplot_ dari variabel `Gender`, terlihat bahwa jumlah pelanggan laki-laki lebih banyak dibandingkan dengan pelanggan perempuan.
- Pada _countplot_ dari variabel `Subscription Type`, terlihat bahwa sebaran pelanggan relatif merata pada masing-masing tipe _subscription_.
- Pada _countplot_ dari variabel `Contract Length`, terlihat bahwa pelanggan lebih banyak yang memiliki kontrak tahunan dan triwulanan dibandingkan dengan kontrak bulanan.

![alt text](https://github.com/mausneg/Customer-Churn-Classification/blob/main/images/image-3.png)

Dari hasil analisis _univariate_ pada _numerical features_ di atas, dapat diamati bahwa:

- Pada histogram dari variabel `Age`, terlihat bahwa sebagian besar pelanggan berusia antara 20-50 tahun.
- Pada histogram dari variabel `Tenure` dan `Usage Frequency`, terlihat bahwa sebaran pelanggan relatif merata.
- Pada histogram dari variabel `Support Calls` terlihat bahwa sebagian besar pelanggan melakukan panggilan dukungan kurang dari 5 kali.
- Pada histogram dari variabel `Payment Delay`, terlihat bahwa sebagian besar pelanggan melakukan keterlambatan pembayaran kurang dari 20 hari.
- Pada histogram dari variabel `Total Spend`, terlihat bahwa sebagian besar pelanggan mengeluarkan biaya lebih dari 500 dolar.
- Pada histogram dari variabel `Last Interaction`, terlihat bahwa sebaran pelanggan relatif merata.

### Multivariate Analysis

Pada analisis _multivariate_, akan dilakukan analisis korelasi antar variabel pada dataset. Hal ini dilakukan untuk mengetahui korelasi antar variabel pada dataset sehingga dapat diketahui variabel mana saja yang memiliki korelasi tinggi dengan label _churn_.

![alt text](https://github.com/mausneg/Customer-Churn-Classification/blob/main/images/image-4.png)

Gambar 4. _Heatmap_ Korelasi _Customer Churn_

Dari Gambar 4 dapat diamati bahwa:

- Hanya variabel `Support Calls` yang memiliki korelasi moderat dengan label _churn_.
- Variabel `Age`, `Payment Delay`, `Last Interaction`,dan `Total Spend` memiliki korelasi rendah dengan label _churn_.
- Sedangkan variabel `Tenure` dan `Usage Frequency` memiliki korelasi mendekati nol dengan label _churn_.

Pada kasus kali ini, karena model yang akan digunakan adalah _deep learning_, maka tidak variabel yang memiliki korelasi rendah dengan label _churn_ tidak dihapus. Hal ini karena _deep learning_ dapat memberikan _weight_ yang tepat untuk setiap variabel sehingga tidak perlu melakukan _feature selection_.

## Data Preparation

Pada tahap ini, perlu dilakukan beberapa proses untuk mempersiapkan data sebelum dilakukan pemodelan. Proses-proses tersebut antara lain:

### Menghapus Kolom yang Tidak Diperlukan

Pada tahap ini, kolom `CustomerID` dihapus karena tidak diperlukan dalam proses pemodelan.

### Encoding Variabel Kategorikal

Pada fitur `Gender` _female_ diubah menjadi 0 dan _male_ diubah menjadi 1, sedangkan pada fitur `Subscription Type` dan `Contract Length` dilakukan _one-hot encoding_.

### Splitting Data

Dataset akan dibagi menjadi data _training_, data _testing_, dan data _validation_. Dengan ukuran data _training_ sebesar 80%, data _testing_ sebesar 10%, dan data _validation_ sebesar 10%. Masing-masing ukuran ini dipertimbangkan dengan ukuran dataset yang ada. Sehingga didapatkan hasil bahwa data _training_ sebesar 404164, data _testing_ sebesar 50521, dan data _validation_ sebesar 50521. Dengan ukuran 50521 pada data _testing_ dan data _validation_, merupakan ukuran yang cukup besar untuk melakukan evaluasi model.

## Modeling

Pada proyek ini, akan digunakan model _machine learning_ tensorflow dengan arsitektur _deep learning_. Hal ini karena beberapa alasan, antara lain:

- Kemampuan untuk menangkap pola yang kompleks: _Deep learning_ mampu mempelajari pola yang sangat kompleks dalam data. Ini berarti bahwa jika ada hubungan non-_linear_ atau interaksi antara fitur dalam _dataset_, model _deep learning_ mungkin mampu menangkapnya.
- Kemampuan untuk menangani data berdimensi tinggi: Karena pada kasus ini memiliki sekitar 14 fitur sehingga model _deep learning_ dapat mengelola kompleksitas ini.
- Pemilihan fitur otomatis: _Deep learning_ dapat mempelajari fitur-fitur penting secara otomatis selama proses pelatihan karena pada kasus ini banyak fitur yang tidak memiliki korelasi yang tinggi dengan label _churn_.
- Performa: Dalam banyak kasus, _deep learning_ telah menunjukkan performa yang sangat baik dalam berbagai tugas prediksi, sering kali melebihi model lain seperti _regresi logistik_ atau _random forest_ [^2^].

Pada proses pelatihan model akan dilakukan _hyperparameter tuning_ dengan _Graduate Student Descent_ (GSD). Hal ini karena GSD merupakan metode yang paling umum digunakan untuk melakukan _hyperparameter tuning_ pada _deep learning_. Berikut merupakan _hyperparameter_ yang akan di _tuning_:

- _Optimizer_: _Optimizer_ mengontrol bagaimana model diperbarui berdasarkan data yang dilihat dan fungsi _loss_-nya. Pada kasus ini, akan dicoba menggunakan _optimizer_ _adam_ dengan _learning rate_ 0.001 (_default_), 0.0001, dan 0.01.
- Jumlah _Neuron_ pada _Hidden Layer_: Jumlah _neuron_ dalam _hidden layer_ dapat mempengaruhi kapasitas model untuk mempelajari pola dalam data. Terlalu sedikit _neuron_ dapat menyebabkan _underfitting_, sementara terlalu banyak _neuron_ dapat menyebabkan _overfitting_. Pada kasus ini, akan dicoba menggunakan 32, 64, 128, 256, 512, dan 1024.
- Jumlah _Hidden Layer_: Jumlah _hidden layer_ dalam model _deep learning_ juga dapat mempengaruhi kapasitas model. Model dengan lebih banyak layer dapat mempelajari pola yang lebih kompleks, tetapi juga lebih berisiko _overfitting_ dan membutuhkan lebih banyak data untuk pelatihan. Pada kasus ini, akan dicoba menggunakan 1 sampai 3 _hidden layer_.
- _Epoch_: _Epoch_ adalah jumlah kali seluruh dataset melalui _neural network_ selama pelatihan. Pada kasus ini, akan dicoba menggunakan 10 dan 20 _epoch_ saja.
- _Batch Size_: _Batch size_ adalah jumlah sampel yang diproses sebelum model diperbarui. _Batch size_ yang lebih kecil dapat menghasilkan pembaruan yang lebih sering, tetapi juga dapat menyebabkan _noise_ dalam pembaruan tersebut. Pada kasus ini, akan dicoba menggunakan _batch size_ 128, 256, 512, dan 1024.

Dari proses _tuning_ maka didapatkan model yang terbaik terdiri dari 1024 _neuron_, _layer_ kedua terdiri dari 512 _neuron_, dan _layer_ terakhir terdiri dari 1 _neuron_. Model ini menggunakan fungsi aktivasi _relu_ pada _layer_ pertama dan kedua, karena _relu_ merupakan fungsi aktivasi yang paling umum digunakan pada _hidden layer_. Sedangkan pada _layer_ terakhir menggunakan fungsi aktivasi _sigmoid_, karena _sigmoid_ merupakan fungsi aktivasi yang paling umum digunakan pada _output_ _layer_ untuk klasifikasi biner.

Selanjutnya, model akan di _compile_ dengan _optimizer_ adam dengan _learning rate default_ (0.001), _loss function binary crossentropy_, dan _metrics accuracy_. _Optimizer_ _adam_ merupakan _optimizer_ yang paling umum digunakan pada _deep learning_ karena dapat melakukan penyesuaian _learning rate_ secara otomatis. _Loss function binary crossentropy_ merupakan _loss function_ yang paling umum digunakan pada klasifikasi biner. _Metrics accuracy_ digunakan untuk mengukur performa model.

## Evaluation

Pada tahap ini, model akan dievaluasi dengan menggunakan data _testing_ dan data _validation_. Evaluasi dilakukan dengan menggunakan metrik _accuracy_. _Accuracy_ sendiri merupakan metrik yang paling umum digunakan untuk mengukur performa model klasifikasi. _Accuracy_ mengukur seberapa sering model membuat prediksi label yang benar dari label sebenarnya. Dengan menggunakan metrik _accuracy_, model akan dievaluasi dengan menggunakan data _testing_ dan data _validation_.

![alt text](https://github.com/mausneg/Customer-Churn-Classification/blob/main/images/image-5.png)

Gambar 5. Diagram Plot Evaluasi Model _Customer Churn_

Dari Gambar 5, dapat diamati bahwa:

- Model memiliki akurasi sekitar 0.92 pada data _training_, 0.92 pada data _testing_, dan 0.92 pada data _validation_.
- Model memiliki akurasi yang konsisten pada data _training_, data _testing_, dan data _validation_ sehingga model dapat dikatakan _goodfit_. Hal ini menunjukkan bahwa model tidak _overfitting_ atau _underfitting_. _Overfitting_ sendiri terjadi ketika model memiliki perbedaan _accuracy_ yang besar antara data _training_ dan data _testing_ atau antara data _testing_ dan data _validation_. Sedangkan _underfitting_ terjadi ketika model memiliki _accuracy_ yang rendah pada data _training_, data _testing_, dan data _validation_.
- Model memiliki performa yang sangat baik dalam memprediksi _churn_ pelanggan dengan akurasi di atas 85%.
- Dengan menggunakan model ini, perusahaan dapat mengidentifikasi pelanggan yang berisiko _churn_ dan mengambil tindakan proaktif untuk mempertahankan mereka. Misalnya, mereka mungkin menawarkan diskon, peningkatan layanan, atau komunikasi pribadi untuk meningkatkan kepuasan dan loyalitas pelanggan.
- Dengan pendekatan ini, model dapat membantu perusahaan mengurangi _churn_ pelanggan, meningkatkan retensi pelanggan, dan pada akhirnya meningkatkan pendapatan dan pertumbuhan bisnis.

## References

[^1^]: [Ahmadi, T., Wulandari, A., & Suhatman, H. Sistem Customer Churn Prediction Menggunakan Machine Learning pada Perusahaan ISP.](https://repository.pnj.ac.id/id/eprint/14345/3/JURNAL.pdf)
[^2^]: [Sofia, R. N., & Supriyadi, D. (2021). Komparasi Metode Machine Learning dan Deep Learning untuk Deteksi Emosi pada Text di Sosial Media. JUPITER (Jurnal Penelitian Ilmu dan Teknik Komputer), 13(2), 130-139.](https://jurnal.polsri.ac.id/index.php/jupiter/article/view/3603/1677)
