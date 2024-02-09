# Laporan Proyek Machine Learning - Maulana Surya Negara

## Domain Proyek

Churn pelanggan merujuk pada fenomena di mana pelanggan menghentikan hubungan atau langganan mereka dengan perusahaan atau penyedia layanan. Ini mewakili tingkat di mana pelanggan berhenti menggunakan produk atau layanan perusahaan dalam periode tertentu. Churn adalah metrik penting bagi bisnis karena berdampak langsung pada pendapatan, pertumbuhan, dan retensi pelanggan.

Dalam konteks dataset Churn, label churn menunjukkan apakah pelanggan telah churn atau tidak. Pelanggan yang churn adalah orang yang telah memutuskan untuk menghentikan langganan atau penggunaan layanan perusahaan. Di sisi lain, pelanggan yang tidak churn adalah orang yang terus terlibat dan mempertahankan hubungan mereka dengan perusahaan.

Memahami churn pelanggan sangat penting bagi bisnis untuk mengidentifikasi pola, faktor, dan indikator yang berkontribusi terhadap attrition pelanggan. Dengan menganalisis perilaku churn dan fitur-fitur yang terkait, perusahaan dapat mengembangkan strategi untuk mempertahankan pelanggan yang ada, meningkatkan kepuasan pelanggan, dan mengurangi turnover pelanggan. Teknik pemodelan prediktif juga dapat diterapkan untuk meramalkan dan secara proaktif menangani churn potensial, memungkinkan perusahaan untuk mengambil langkah-langkah proaktif untuk mempertahankan pelanggan yang berisiko.

## Business Understanding

### Problem Statements

- Perusahaan mengalami tingkat churn pelanggan yang tinggi, yang berdampak negatif pada pendapatan dan pertumbuhan mereka.
- Perusahaan saat ini tidak memiliki cara untuk mengidentifikasi pelanggan yang berisiko churn.
- Tanpa kemampuan untuk memprediksi churn, perusahaan tidak dapat mengambil tindakan proaktif untuk mempertahankan pelanggan.

### Goals

- Membangun model prediktif yang dapat memprediksi churn pelanggan dengan akurasi yang tinggi.
- Mengidentifikasi fitur-fitur yang paling berkontribusi terhadap churn pelanggan.
- Menggunakan model untuk mengidentifikasi pelanggan yang berisiko churn.
- Mengambil tindakan proaktif berdasarkan prediksi model untuk mempertahankan pelanggan dan mengurangi churn.
- Meningkatkan retensi pelanggan dan, pada akhirnya, pendapatan dan pertumbuhan perusahaan.

## Data Understanding

Data yang digunakan pada proyek kali ini adalah data yang diambil dari <a href='https://www.kaggle.com/datasets/nelgiriyewithana/apple-quality' target='_blank'>Kaggle</a>. Trainning file untuk dataset CHURN berisi kumpulan 440882 record pelanggan beserta fitur dan label churn mereka masing-masing. File ini berfungsi sebagai sumber utama untuk melatih model machine learning untuk memprediksi churn pelanggan. Setiap record dalam file training mewakili pelanggan dan mencakup fitur seperti CustomerID, Age Gender, Tenure, Usage Frequency, Support Calls, Payment Delay, Subscription Type, Contract Length, Total Spend, dan Last Interaction. Label churn menunjukkan apakah pelanggan telah churn (1) atau tidak (0). Dengan memanfaatkan file pelatihan ini, bisnis dapat mengembangkan model prediksi churn yang akurat untuk mengidentifikasi pelanggan yang paling mungkin untuk churn dan mengambil tindakan proaktif untuk mempertahankannya.

### Variabel-variabel pada Customer Churn dataset adalah sebagai berikut:

- `CustomerID`: ID unik untuk setiap pelanggan.
- `Age`: Usia pelanggan.
- `Gender`: Jenis kelamin pelanggan.
- `Tenure`: Masa aktif pelanggan dalam bulan.
- `Usage Frequency`: Frekuensi penggunaan layanan dalam sebulan.
- `Support Calls`: Jumlah panggilan dukungan pelanggan.
- `Payment Delay`: Jumlah hari keterlambatan pembayaran.
- `Subscription Type`: Tipe langganan pelanggan.
- `Contract Length`: Durasi kontrak pelanggan dalam bulan.
- `Total Spend`: Total pengeluaran pelanggan.
- `Last Interaction`: Waktu interaksi terakhir dengan pelanggan.
- `Churn`: Label churn pelanggan (1: churn, 0: tidak churn).

## Data Preparation

Pada tahap ini, perlu dilakukan beberapa proses untuk mempersiapkan data sebelum dilakukan pemodelan. Proses-proses tersebut antara lain:
- Melakukan pengecekan terhadap tipe data dari setiap kolom pada dataset. Hal ini dilakukan untuk memastikan bahwa tipe data dari setiap kolom sudah sesuai dengan yang diharapkan.
- Melakukan pengecekan terhadap missing value pada dataset. Hal ini dilakukan untuk memastikan bahwa tidak ada missing value pada dataset.
- Melakukan pengecekan terhadap duplikasi data pada dataset. Hal ini dilakukan untuk memastikan bahwa tidak ada duplikasi data pada dataset.
- Melakukan pengecekan terhadap outlier pada dataset dengan menggunakan describe statistics. Hal ini dilakukan untuk memastikan bahwa tidak ada outlier pada dataset sehingga tidak mempengaruhi hasil dari model machine learning yang akan dibuat.
- Mengatasi missing value pada dataset. Hal ini dilakukan dengan cara mengisi missing value dengan nilai yang sesuai atau menghapus baris yang memiliki missing value karena jumlah missing value yang sedikit.
- Melakukan pengecekan terhadap korelasi antar variabel pada dataset. Hal ini dilakukan untuk mengetahui korelasi antar variabel pada dataset sehingga dapat diketahui variabel mana saja yang memiliki korelasi tinggi dengan label churn.
- Menghapus kolom yang tidak diperlukan dan yang memiliki korelasi sangat rendah dengan label churn. Hal ini dilakukan untuk mempercepat proses pemodelan dan mengurangi kompleksitas model.
- Melakukan encoding terhadap kolom-kolom kategorikal pada dataset. Hal ini dilakukan untuk mengubah tipe data dari kolom-kolom kategorikal menjadi numerik sehingga dapat digunakan pada model machine learning.
- Melakukan split data menjadi data train, test, dan validation. Hal ini dilakukan untuk membagi data menjadi data train, test, dan validation dengan proporsi tertentu sehingga dapat digunakan untuk melatih model, menguji model, dan mengevaluasi model.

## Modeling

Tahapan ini membahas mengenai model machine learning yang digunakan untuk menyelesaikan permasalahan. Anda perlu menjelaskan tahapan dan parameter yang digunakan pada proses pemodelan.

**Rubrik/Kriteria Tambahan (Opsional)**:

- Menjelaskan kelebihan dan kekurangan dari setiap algoritma yang digunakan.
- Jika menggunakan satu algoritma pada solution statement, lakukan proses improvement terhadap model dengan hyperparameter tuning. **Jelaskan proses improvement yang dilakukan**.
- Jika menggunakan dua atau lebih algoritma pada solution statement, maka pilih model terbaik sebagai solusi. **Jelaskan mengapa memilih model tersebut sebagai model terbaik**.

## Evaluation

Pada bagian ini anda perlu menyebutkan metrik evaluasi yang digunakan. Lalu anda perlu menjelaskan hasil proyek berdasarkan metrik evaluasi yang digunakan.

Sebagai contoh, Anda memiih kasus klasifikasi dan menggunakan metrik **akurasi, precision, recall, dan F1 score**. Jelaskan mengenai beberapa hal berikut:

- Penjelasan mengenai metrik yang digunakan
- Menjelaskan hasil proyek berdasarkan metrik evaluasi

Ingatlah, metrik evaluasi yang digunakan harus sesuai dengan konteks data, problem statement, dan solusi yang diinginkan.

**Rubrik/Kriteria Tambahan (Opsional)**:

- Menjelaskan formula metrik dan bagaimana metrik tersebut bekerja.

**---Ini adalah bagian akhir laporan---**

_Catatan:_

- _Anda dapat menambahkan gambar, kode, atau tabel ke dalam laporan jika diperlukan. Temukan caranya pada contoh dokumen markdown di situs editor [Dillinger](https://dillinger.io/), [Github Guides: Mastering markdown](https://guides.github.com/features/mastering-markdown/), atau sumber lain di internet. Semangat!_
- Jika terdapat penjelasan yang harus menyertakan code snippet, tuliskan dengan sewajarnya. Tidak perlu menuliskan keseluruhan kode project, cukup bagian yang ingin dijelaskan saja.
