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

Semua poin di atas harus diuraikan dengan jelas. Anda bebas menuliskan berapa pernyataan masalah dan juga goals yang diinginkan.

**Rubrik/Kriteria Tambahan (Opsional)**:

- Menambahkan bagian “Solution Statement” yang menguraikan cara untuk meraih goals. Bagian ini dibuat dengan ketentuan sebagai berikut:

  ### Solution statements

  - Mengajukan 2 atau lebih solution statement. Misalnya, menggunakan dua atau lebih algoritma untuk mencapai solusi yang diinginkan atau melakukan improvement pada baseline model dengan hyperparameter tuning.
  - Solusi yang diberikan harus dapat terukur dengan metrik evaluasi.

## Data Understanding

Paragraf awal bagian ini menjelaskan informasi mengenai data yang Anda gunakan dalam proyek. Sertakan juga sumber atau tautan untuk mengunduh dataset. Contoh: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Restaurant+%26+consumer+data).

Selanjutnya uraikanlah seluruh variabel atau fitur pada data. Sebagai contoh:

### Variabel-variabel pada Restaurant UCI dataset adalah sebagai berikut:

- accepts : merupakan jenis pembayaran yang diterima pada restoran tertentu.
- cuisine : merupakan jenis masakan yang disajikan pada restoran.
- dst

**Rubrik/Kriteria Tambahan (Opsional)**:

- Melakukan beberapa tahapan yang diperlukan untuk memahami data, contohnya teknik visualisasi data atau exploratory data analysis.

## Data Preparation

Pada bagian ini Anda menerapkan dan menyebutkan teknik data preparation yang dilakukan. Teknik yang digunakan pada notebook dan laporan harus berurutan.

**Rubrik/Kriteria Tambahan (Opsional)**:

- Menjelaskan proses data preparation yang dilakukan
- Menjelaskan alasan mengapa diperlukan tahapan data preparation tersebut.

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
