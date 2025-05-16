# Proyek Akhir: Menyelesaikan Permasalahan Perusahaan Edutech

## Business Understanding

Jaya Jaya Institut adalah institusi pendidikan perguruan tinggi ternama yang telah beroperasi sejak tahun 2000 dan dikenal karena reputasi baik dalam mencetak lulusan berkualitas. Namun, institusi ini menghadapi tantangan signifikan akibat tingginya tingkat dropout siswa, yang mencapai 32,12% dari total 4.424 siswa. Tujuan bisnis utama adalah mempertahankan siswa dan meningkatkan tingkat kelulusan dengan mendeteksi sejak dini siswa yang berisiko dropout untuk diberikan bimbingan khusus, sehingga reputasi dan keberlanjutan institusi dapat dipertahankan.

### Permasalahan Bisnis

- Tingginya tingkat dropout siswa (32,12% atau 1.421 dari 4.424 siswa) yang berdampak negatif pada reputasi dan pendapatan institusi.
- Kesulitan dalam mengidentifikasi siswa berisiko dropout secara dini untuk intervensi tepat waktu.
- Faktor akademik (misalnya, rendahnya unit mata kuliah yang disetujui) dan finansial (misalnya, tunggakan biaya kuliah) yang berkontribusi pada dropout.
- Kurangnya pemahaman mendalam tentang faktor demografis dan non-akademik yang memengaruhi keputusan dropout.

### Cakupan Proyek

Proyek ini mencakup pengembangan solusi berbasis data untuk mengatasi permasalahan dropout di Jaya Jaya Institut, dengan fokus pada:

- Analisis dataset kinerja siswa untuk memahami pola dropout.
- Pembuatan dashboard interaktif untuk memantau performa siswa berdasarkan status, faktor akademik, finansial, dan demografis.
- Pengembangan prototipe sistem machine learning berbasis XGBoost untuk memprediksi siswa berisiko dropout.
- Penyediaan rekomendasi aksi berbasis data untuk intervensi strategis.

### Persiapan

**Sumber data:**  
Dataset kinerja siswa yang disediakan oleh Jaya Jaya Institut, dapat diunduh melalui tautan: [students' performance](https://example.com/students-performance.csv). Data mencakup informasi seperti status siswa (Dropout, Enrolled, Graduate), unit mata kuliah, biaya kuliah, usia pendaftaran, dan faktor ekonomi.

**Setup Environment:**  

1. **Clone repository**  

    ```bash
    git clone https://github.com/username/repo-jaya-jaya.git
    cd repo-jaya-jaya
    ```

2. **Buat virtual environment**  

    ```bash
    python -m venv venv
    source venv/bin/activate  # Untuk Linux/Mac
    venv\Scripts\activate     # Untuk Windows
    ```

3. **Install dependencies**  

    ```bash
    pip install -r requirements.txt
    ```

4. **Jalankan Jupyter Notebook**  

    ```bash
    jupyter notebook
    ```

5. **Pastikan file data tersedia**  
    Simpan file dataset (`students-performance.csv`) di folder proyek.

---

## Business Dashboard

Dashboard yang telah dibuat memberikan gambaran komprehensif tentang performa siswa Jaya Jaya Institut, mencakup:

- Total siswa (4.424), persentase dropout (32,12%), dan rata-rata unit mata kuliah disetujui semester pertama (4,71).
- Visualisasi "Count of Status" menunjukkan distribusi siswa (49% Graduate, 17,9% Enrolled, 32,12% Dropout).
- "Academic Factor" menampilkan perbandingan unit mata kuliah disetujui antara Dropout, Enrolled, dan Graduate.
- "Financial Factor" menggambarkan status pembayaran biaya kuliah dan utang siswa.
- "Demographic Factor" menganalisis usia pendaftaran dan gender berdasarkan status siswa.

Dashboard ini dapat diakses melalui docker atau melihat langsung melalui hasil screenshot dashboard

- **Kredensial Akses:**
  - Email: [zidirsyadin@gmail.com]
  - Password: [accbang123]

---

## Menjalankan Sistem Machine Learning

Prototipe sistem machine learning menggunakan model XGBoost untuk memprediksi status siswa (Dropout, Enrolled, Graduate) dengan akurasi 76,2% pada data test.

**Langkah menjalankan prototipe:**

1. Pastikan environment telah diatur dan dependensi terinstal (lihat bagian Persiapan).
2. Muat dataset menggunakan perintah:

    ```python
    import pandas as pd
    df = pd.read_csv('students-performance.csv', sep=';')
    ```
3. Jalankan skrip pelatihan model dan evaluasi yang tersedia dalam notebook Jupyter.
4. Gunakan model tersimpan (`best_model_xgboost.joblib`) untuk prediksi baru:
    ```python
    import joblib
    loaded_model = joblib.load('best_model_xgboost.joblib')
    ```
Prototipe dapat diakses dan diuji melalui [link prototipe](https://example.com/ml-prototype-jaya-jaya).

---

## Conclusion
Proyek ini berhasil mengidentifikasi pola dropout melalui analisis data dan dashboard interaktif, serta mengembangkan model prediktif dengan akurasi 76,2% untuk mendeteksi siswa berisiko. Meskipun model menunjukkan performa baik pada kelas mayoritas (Graduate), sensitivitas terhadap kelas Dropout dan Enrolled masih perlu ditingkatkan. Dengan implementasi rekomendasi aksi, Jaya Jaya Institut dapat mengurangi tingkat dropout, meningkatkan retensi siswa, dan mendukung keberhasilan akademik secara keseluruhan.

---

### Rekomendasi Action Items
Beberapa rekomendasi action items yang dapat dilakukan perusahaan untuk menyelesaikan permasalahan dan mencapai target:

1. **Identifikasi Dini Siswa Berisiko Dropout**  
    - Gunakan model XGBoost untuk mendeteksi siswa berisiko dropout.  
    - Tingkatkan sensitivitas model terhadap kelas "Dropout" melalui tuning parameter dan teknik balancing data.  
    - Perbarui model secara berkala dengan data terbaru untuk menjaga akurasi.

2. **Dukung Performa Akademik**  
    - Berikan bimbingan akademik (tutor, mentoring) bagi siswa yang lulus kurang dari 5 unit mata kuliah di semester pertama.  
    - Buat sistem peringatan dini untuk siswa dengan performa akademik rendah.

3. **Atasi Kendala Keuangan**  
    - Perluas program beasiswa dan tawarkan rencana pembayaran fleksibel bagi siswa yang mengalami kesulitan biaya.  
    - Sediakan layanan konseling keuangan untuk membantu siswa mengelola keuangan dan utang.

4. **Intervensi Berbasis Demografi**  
    - Dukung siswa usia 30-45 tahun dengan opsi kuliah malam atau daring, serta program orientasi khusus.  
    - Selidiki alasan tingginya dropout pada siswa laki-laki dan buat program pendukung seperti mentoring khusus.

5. **Manfaatkan Dashboard untuk Pemantauan**  
    - Perbarui dashboard secara berkala untuk memantau tren dropout dan performa siswa.  
    - Latih staf untuk menggunakan dashboard dalam mengidentifikasi siswa berisiko dan mengevaluasi efektivitas intervensi.

6. **Kumpulkan Data Kualitatif**  
    - Lakukan survei atau wawancara untuk memahami faktor non-akademik (misal: stres, motivasi) yang memengaruhi dropout.  
    - Tambahkan fitur baru (kehadiran, aktivitas ekstrakurikuler) ke model untuk meningkatkan akurasi prediksi.
