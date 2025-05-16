import streamlit as st
import pandas as pd
import plotly.express as px
from sqlalchemy import create_engine
import urllib.parse
import joblib
from sklearn.preprocessing import LabelEncoder
import numpy as np
import os

# Konfigurasi halaman Streamlit
st.set_page_config(page_title="Jaya Jaya Institut Dropout Dashboard", layout="wide")

# Judul dashboard
st.title("Jaya Jaya Institut - Dashboard Pemantauan Dropout Siswa")

# Koneksi ke Supabase
st.sidebar.header("Koneksi Database")
try:
    password = "Ichaz781anakjokam*"
    encoded_password = urllib.parse.quote(password)
    URL = f"postgresql://postgres.mybfsltcugisqcirtxay:{encoded_password}@aws-0-ap-southeast-1.pooler.supabase.com:6543/postgres"
    engine = create_engine(URL)
    df = pd.read_sql("SELECT * FROM studentperformance", engine)
    st.sidebar.success("Berhasil terhubung ke Supabase!")
except Exception as e:
    st.sidebar.error(f"Gagal terhubung ke Supabase: {str(e)}")
    st.stop()

# Memuat model dan label encoder

try:
    model_path = os.path.join(os.path.dirname(__file__), "model/best_model_xgboost.joblib")
    model = joblib.load(model_path)
    le = LabelEncoder()
    le.classes_ = np.array(['Dropout', 'Enrolled', 'Graduate'])
except Exception as e:
    st.error(f"Gagal memuat model: {str(e)}")
    st.stop()

# Sidebar untuk filter
st.sidebar.header("Filter Data")
age_filter = st.sidebar.slider("Usia saat Pendaftaran", int(df['Age_at_enrollment'].min()), int(df['Age_at_enrollment'].max()), (18, 50))
tuition_filter = st.sidebar.selectbox("Biaya Kuliah Lunas", ["Semua", "Ya", "Tidak"], index=0)
status_filter = st.sidebar.multiselect("Status", options=['Dropout', 'Enrolled', 'Graduate'], default=['Dropout', 'Enrolled', 'Graduate'])

# Filter DataFrame
filtered_df = df[
    (df['Age_at_enrollment'].between(age_filter[0], age_filter[1])) &
    (df['Status'].isin(status_filter))
]
if tuition_filter != "Semua":
    filtered_df = filtered_df[filtered_df['Tuition_fees_up_to_date'] == (1 if tuition_filter == "Ya" else 0)]

# Tab untuk navigasi
tab1, tab2, tab3 = st.tabs(["Ringkasan", "Analisis Faktor", "Prediksi Dropout"])

# Tab 1: Ringkasan
with tab1:
    st.header("Ringkasan Statistik")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Siswa", len(filtered_df))
    with col2:
        st.metric("Jumlah Dropout", len(filtered_df[filtered_df['Status'] == 'Dropout']))
    with col3:
        st.metric("Jumlah Lulus", len(filtered_df[filtered_df['Status'] == 'Graduate']))

    st.subheader("Distribusi Status Siswa")
    status_counts = filtered_df['Status'].value_counts().reset_index()
    status_counts.columns = ['Status', 'Jumlah']
    fig_pie = px.pie(status_counts, names='Status', values='Jumlah', title="Distribusi Status Siswa")
    st.plotly_chart(fig_pie, use_container_width=True)

# Tab 2: Analisis Faktor
with tab2:
    st.header("Analisis Faktor Risiko Dropout")

    # Visualisasi 1: Unit disetujui semester 1
    st.subheader("Unit Disetujui Semester 1 Berdasarkan Status")
    fig_bar = px.histogram(
        filtered_df,
        x='Curricular_units_1st_sem_approved',
        color='Status',
        barmode='group',
        title="Distribusi Unit Disetujui Semester 1",
        labels={'Curricular_units_1st_sem_approved': 'Unit Disetujui Semester 1'}
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # Visualisasi 2: Nilai vs Unit disetujui
    st.subheader("Hubungan Nilai dan Unit Disetujui")
    fig_scatter = px.scatter(
        filtered_df,
        x='Curricular_units_1st_sem_grade',
        y='Curricular_units_1st_sem_approved',
        color='Status',
        title="Nilai vs Unit Disetujui (Semester 1)",
        labels={'Curricular_units_1st_sem_grade': 'Nilai Semester 1', 'Curricular_units_1st_sem_approved': 'Unit Disetujui'}
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    # Visualisasi 3: Biaya kuliah lunas
    st.subheader("Status Biaya Kuliah Berdasarkan Status")
    tuition_status = filtered_df.groupby(['Tuition_fees_up_to_date', 'Status']).size().reset_index(name='Jumlah')
    tuition_status['Tuition_fees_up_to_date'] = tuition_status['Tuition_fees_up_to_date'].map({1: 'Lunas', 0: 'Belum Lunas'})
    fig_tuition = px.bar(
        tuition_status,
        x='Status',
        y='Jumlah',
        color='Tuition_fees_up_to_date',
        barmode='stack',
        title="Status Biaya Kuliah per Status Siswa",
        labels={'Tuition_fees_up_to_date': 'Status Biaya Kuliah'}
    )
    st.plotly_chart(fig_tuition, use_container_width=True)

    # Visualisasi 4: Distribusi usia
    st.subheader("Distribusi Usia Berdasarkan Status")
    fig_box = px.box(
        filtered_df,
        x='Status',
        y='Age_at_enrollment',
        title="Distribusi Usia Saat Pendaftaran",
        labels={'Age_at_enrollment': 'Usia'}
    )
    st.plotly_chart(fig_box, use_container_width=True)

# Tab 3: Prediksi Dropout
with tab3:
    st.header("Prediksi Risiko Dropout")
    st.write("Masukkan data siswa untuk memprediksi risiko dropout. Silakan baca penjelasan fitur di bawah ini untuk memahami makna setiap kolom sebelum mengisi form.")

    # Penjelasan Fitur dalam Expander
    with st.expander("Penjelasan Fitur untuk Penginputan"):
        st.markdown("""
        Berikut adalah penjelasan untuk setiap fitur yang diperlukan dalam form prediksi. Harap masukkan nilai sesuai dengan kategori atau rentang yang ditentukan agar prediksi lebih akurat.

        - **Marital Status**: Status perkawinan siswa. Kategori: 1 (lajang), 2 (menikah), 3 (janda), 4 (cerai), 5 (persatuan faktual), 6 (pisah secara hukum).
        - **Application Mode**: Metode aplikasi siswa. Kategori: 1 (kontingen umum fase 1), 2 (Ordinance No. 612/93), 5 (kontingen khusus fase 1 - Pulau Azores), 7 (pemegang kursus tinggi lainnya), 10 (Ordinance No. 854-B/99), 15 (pelajar internasional - sarjana), 16 (kontingen khusus fase 1 - Pulau Madeira), 17 (kontingen umum fase 2), 18 (kontingen umum fase 3), 26 (Ordinance No. 533-A/99, item b2 - Rencana Berbeda), 27 (Ordinance No. 533-A/99, item b3 - Institusi Lain), 39 (usia di atas 23 tahun), 42 (transfer), 43 (ganti kursus), 44 (pemegang diploma spesialisasi teknologi), 51 (ganti institusi/kursus), 53 (pemegang diploma siklus pendek), 57 (ganti institusi/kursus - internasional).
        - **Application Order**: Urutan aplikasi siswa. Nilai numerik antara 0 (pilihan pertama) hingga 9 (pilihan terakhir).
        - **Course**: Kursus yang diambil siswa. Kategori: 33 (Teknologi Produksi Bahan Bakar Bio), 171 (Desain Animasi dan Multimedia), 8014 (Layanan Sosial - kehadiran malam), 9003 (Agronomi), 9070 (Desain Komunikasi), 9085 (Keperawatan Hewan), 9119 (Teknik Informatika), 9130 (Equinculture), 9147 (Manajemen), 9238 (Layanan Sosial), 9254 (Pariwisata), 9500 (Keperawatan), 9556 (Kebersihan Mulut), 9670 (Manajemen Periklanan dan Pemasaran), 9773 (Jurnalisme dan Komunikasi), 9853 (Pendidikan Dasar), 9991 (Manajemen - kehadiran malam).
        - **Daytime/Evening Attendance**: Kehadiran siswa pada siang atau malam hari. Kategori: 1 (siang), 0 (malam).
        - **Previous Qualification**: Kualifikasi sebelumnya siswa. Kategori: 1 (pendidikan menengah), 2 (sarjana), 3 (gelar tinggi), 4 (master), 5 (doktor), 6 (frekuensi pendidikan tinggi), 9 (tahun ke-12 sekolah - belum selesai), 10 (tahun ke-11 sekolah - belum selesai), 12 (lainnya - tahun ke-11 sekolah), 14 (tahun ke-10 sekolah), 15 (tahun ke-10 sekolah - belum selesai), 19 (pendidikan dasar siklus ke-3 atau setara), 38 (pendidikan dasar siklus ke-2 atau setara), 39 (kursus spesialisasi teknologi), 40 (gelar tinggi - siklus 1), 42 (kursus teknis tinggi profesional), 43 (master - siklus 2).
        - **Previous Qualification Grade**: Nilai kualifikasi sebelumnya siswa. Nilai numerik antara 0 hingga 200.
        - **Nationality**: Kebangsaan siswa. Kategori: 1 (Portugis), 2 (Jerman), 6 (Spanyol), 11 (Italia), 13 (Belanda), 14 (Inggris), 17 (Lituania), 21 (Angola), 22 (Tanjung Verde), 24 (Guinea), 25 (Mozambik), 26 (Santome), 32 (Turki), 41 (Brasil), 62 (Rumania), 100 (Moldova), 101 (Meksiko), 103 (Ukraina), 105 (Rusia), 108 (Kuba), 109 (Kolombia).
        - **Mother's Qualification**: Kualifikasi ibu siswa. Kategori: 1 (pendidikan menengah - tahun ke-12 atau setara), 2 (sarjana), 3 (gelar tinggi), 4 (master), 5 (doktor), 6 (frekuensi pendidikan tinggi), 9 (tahun ke-12 sekolah - belum selesai), 10 (tahun ke-11 sekolah - belum selesai), 11 (tahun ke-7 lama), 12 (lainnya - tahun ke-11 sekolah), 14 (tahun ke-10 sekolah), 18 (kursus perdagangan umum), 19 (pendidikan dasar siklus ke-3 atau setara), 22 (kursus teknis-profesional), 26 (tahun ke-7 sekolah), 27 (siklus ke-2 kursus SMA umum), 29 (tahun ke-9 sekolah - belum selesai), 30 (tahun ke-8 sekolah), 34 (tidak diketahui), 35 (tidak bisa membaca atau menulis), 36 (bisa membaca tanpa 4 tahun sekolah), 37 (pendidikan dasar siklus ke-1 atau setara), 38 (pendidikan dasar siklus ke-2 atau setara), 39 (kursus spesialisasi teknologi), 40 (gelar tinggi - siklus 1), 41 (kursus studi tinggi khusus), 42 (kursus teknis tinggi profesional), 43 (master - siklus 2), 44 (doktor - siklus 3).
        - **Father's Qualification**: Kualifikasi ayah siswa. Kategori: 1 (pendidikan menengah - tahun ke-12 atau setara), 2 (sarjana), 3 (gelar tinggi), 4 (master), 5 (doktor), 6 (frekuensi pendidikan tinggi), 9 (tahun ke-12 sekolah - belum selesai), 10 (tahun ke-11 sekolah - belum selesai), 11 (tahun ke-7 lama), 12 (lainnya - tahun ke-11 sekolah), 13 (tahun ke-2 kursus pelengkap SMA), 14 (tahun ke-10 sekolah), 18 (kursus perdagangan umum), 19 (pendidikan dasar siklus ke-3 atau setara), 20 (kursus pelengkap SMA), 22 (kursus teknis-profesional), 25 (kursus pelengkap SMA - belum selesai), 26 (tahun ke-7 sekolah), 27 (siklus ke-2 kursus SMA umum), 29 (tahun ke-9 sekolah - belum selesai), 30 (tahun ke-8 sekolah), 31 (kursus umum administrasi dan perdagangan), 33 (akuntansi dan administrasi tambahan), 34 (tidak diketahui), 35 (tidak bisa membaca atau menulis), 36 (bisa membaca tanpa 4 tahun sekolah), 37 (pendidikan dasar siklus ke-1 atau setara), 38 (pendidikan dasar siklus ke-2 atau setara), 39 (kursus spesialisasi teknologi), 40 (gelar tinggi - siklus 1), 41 (kursus studi tinggi khusus), 42 (kursus teknis tinggi profesional), 43 (master - siklus 2), 44 (doktor - siklus 3).
        - **Mother's Occupation**: Pekerjaan ibu siswa. Kategori: 0 (pelajar), 1 (pejabat legislatif/eksekutif), 2 (spesialis intelektual/ilmiah), 3 (teknisi tingkat menengah), 4 (staf administrasi), 5 (pekerja jasa, keamanan, penjual), 6 (petani/pekerja terampil pertanian), 7 (pekerja terampil industri/konstruksi), 8 (operator mesin/perakitan), 9 (pekerja tidak terampil), 10 (profesi angkatan bersenjata), 90 (situasi lain), 99 (kosong), 122 (profesional kesehatan), 123 (guru), 125 (spesialis ICT), 131 (teknisi sains/teknik tingkat menengah), 132 (teknisi kesehatan tingkat menengah), 134 (teknisi layanan hukum/sosial/olahraga), 141 (pekerja kantor/sekretaris), 143 (operator data/akuntansi/statistik), 144 (staf administrasi lainnya), 151 (pekerja jasa pribadi), 152 (penjual), 153 (pekerja perawatan pribadi), 171 (pekerja konstruksi terampil), 173 (pekerja terampil percetakan/instrumen presisi), 175 (pekerja industri makanan/kayu/pakaian), 191 (pekerja kebersihan), 192 (pekerja tidak terampil pertanian/peternakan), 193 (pekerja tidak terampil industri ekstraktif/konstruksi), 194 (asisten persiapan makanan).
        - **Father's Occupation**: Pekerjaan ayah siswa. Kategori: 0 (pelajar), 1 (pejabat legislatif/eksekutif), 2 (spesialis intelektual/ilmiah), 3 (teknisi tingkat menengah), 4 (staf administrasi), 5 (pekerja jasa, keamanan, penjual), 6 (petani/pekerja terampil pertanian), 7 (pekerja terampil industri/konstruksi), 8 (operator mesin/perakitan), 9 (pekerja tidak terampil), 10 (profesi angkatan bersenjata), 90 (situasi lain), 99 (kosong), 101 (perwira angkatan bersenjata), 102 (sersan angkatan bersenjata), 103 (personel angkatan bersenjata lainnya), 112 (direktur layanan administrasi/komersial), 114 (direktur hotel/katering/perdagangan), 121 (spesialis sains fisik/matematika/teknik), 122 (profesional kesehatan), 123 (guru), 124 (spesialis keuangan/akuntansi/organisasi administrasi), 131 (teknisi sains/teknik tingkat menengah), 132 (teknisi kesehatan tingkat menengah), 134 (teknisi layanan hukum/sosial/olahraga), 135 (teknisi ICT), 141 (pekerja kantor/sekretaris), 143 (operator data/akuntansi/statistik), 144 (staf administrasi lainnya), 151 (pekerja jasa pribadi), 152 (penjual), 153 (pekerja perawatan pribadi), 154 (personel keamanan), 161 (petani/pekerja terampil pertanian), 163 (petani/peternak/pemburu subsisten), 171 (pekerja konstruksi terampil), 172 (pekerja terampil metalurgi), 174 (pekerja terampil listrik/elektronik), 175 (pekerja industri makanan/kayu/pakaian), 181 (operator pabrik/mesin tetap), 182 (pekerja perakitan), 183 (pengemudi kendaraan/peralatan bergerak), 192 (pekerja tidak terampil pertanian/peternakan), 193 (pekerja tidak terampil industri ekstraktif/konstruksi), 194 (asisten persiapan makanan), 195 (pedagang kaki lima non-makanan).
        - **Admission Grade**: Nilai penerimaan siswa. Nilai numerik antara 0 hingga 200.
        - **Displaced**: Apakah siswa merupakan orang yang terlantar. Kategori: 1 (ya), 0 (tidak).
        - **Educational Special Needs**: Apakah siswa memiliki kebutuhan pendidikan khusus. Kategori: 1 (ya), 0 (tidak).
        - **Debtor**: Apakah siswa memiliki utang. Kategori: 1 (ya), 0 (tidak).
        - **Tuition Fees Up to Date**: Apakah biaya kuliah siswa sudah lunas. Kategori: 1 (ya), 0 (tidak).
        - **Gender**: Jenis kelamin siswa. Kategori: 1 (pria), 0 (wanita).
        - **Scholarship Holder**: Apakah siswa pemegang beasiswa. Kategori: 1 (ya), 0 (tidak).
        - **Age at Enrollment**: Usia siswa saat mendaftar. Nilai numerik, biasanya antara 17 hingga 70.
        - **International**: Apakah siswa merupakan pelajar internasional. Kategori: 1 (ya), 0 (tidak).
        - **Curricular Units 1st Sem (Credited)**: Jumlah unit kurikuler yang dikreditkan pada semester 1. Nilai numerik, biasanya antara 0 hingga 20.
        - **Curricular Units 1st Sem (Enrolled)**: Jumlah unit kurikuler yang didaftarkan pada semester 1. Nilai numerik, biasanya antara 0 hingga 20.
        - **Curricular Units 1st Sem (Evaluations)**: Jumlah unit kurikuler yang dievaluasi pada semester 1. Nilai numerik, biasanya antara 0 hingga 20.
        - **Curricular Units 1st Sem (Approved)**: Jumlah unit kurikuler yang disetujui pada semester 1. Nilai numerik, biasanya antara 0 hingga 20.
        - **Curricular Units 1st Sem (Grade)**: Nilai rata-rata unit kurikuler pada semester 1. Nilai numerik antara 0 hingga 20.
        - **Curricular Units 1st Sem (Without Evaluations)**: Jumlah unit kurikuler tanpa evaluasi pada semester 1. Nilai numerik, biasanya antara 0 hingga 20.
        - **Curricular Units 2nd Sem (Credited)**: Jumlah unit kurikuler yang dikreditkan pada semester 2. Nilai numerik, biasanya antara 0 hingga 20.
        - **Curricular Units 2nd Sem (Enrolled)**: Jumlah unit kurikuler yang didaftarkan pada semester 2. Nilai numerik, biasanya antara 0 hingga 20.
        - **Curricular Units 2nd Sem (Evaluations)**: Jumlah unit kurikuler yang dievaluasi pada semester 2. Nilai numerik, biasanya antara 0 hingga 20.
        - **Curricular Units 2nd Sem (Approved)**: Jumlah unit kurikuler yang disetujui pada semester 2. Nilai numerik, biasanya antara 0 hingga 20.
        - **Curricular Units 2nd Sem (Grade)**: Nilai rata-rata unit kurikuler pada semester 2. Nilai numerik antara 0 hingga 20.
        - **Curricular Units 2nd Sem (Without Evaluations)**: Jumlah unit kurikuler tanpa evaluasi pada semester 2. Nilai numerik, biasanya antara 0 hingga 20.
        - **Unemployment Rate**: Tingkat pengangguran pada periode tersebut. Nilai numerik, biasanya antara 0 hingga 20 (berdasarkan data historis).
        - **Inflation Rate**: Tingkat inflasi pada periode tersebut. Nilai numerik, biasanya antara -2 hingga 5 (berdasarkan data historis).
        - **GDP**: Produk Domestik Bruto pada periode tersebut. Nilai numerik, biasanya antara -5 hingga 5 (berdasarkan data historis).
        """)

    # Form input untuk prediksi (diperluas untuk 36 fitur)
    with st.form("prediksi_form"):
        st.subheader("Data Siswa")
        col1, col2, col3 = st.columns(3)

        with col1:
            marital_status = st.number_input("Status Perkawinan", min_value=1, max_value=6, value=1)
            application_mode = st.number_input("Mode Aplikasi", min_value=1, max_value=57, value=1)
            application_order = st.number_input("Urutan Aplikasi", min_value=0, max_value=9, value=0)
            course = st.number_input("Kode Kursus", min_value=0, max_value=9999, value=171)
            daytime_attendance = st.selectbox("Kehadiran Siang/Malam", [1, 0], format_func=lambda x: "Siang" if x == 1 else "Malam")
            previous_qualification = st.number_input("Kualifikasi Sebelumnya", min_value=1, max_value=43, value=1)
            previous_qualification_grade = st.number_input("Nilai Kualifikasi Sebelumnya", min_value=0, max_value=200, value=100)
            nationality = st.number_input("Kebangsaan", min_value=1, max_value=109, value=1)
            mothers_qualification = st.number_input("Kualifikasi Ibu", min_value=1, max_value=44, value=1)
            fathers_qualification = st.number_input("Kualifikasi Ayah", min_value=1, max_value=44, value=1)
            mothers_occupation = st.number_input("Pekerjaan Ibu", min_value=0, max_value=194, value=0)
            fathers_occupation = st.number_input("Pekerjaan Ayah", min_value=0, max_value=195, value=0)
            admission_grade = st.number_input("Nilai Penerimaan", min_value=0, max_value=200, value=100)

        with col2:
            displaced = st.selectbox("Terlantar", [1, 0], format_func=lambda x: "Ya" if x == 1 else "Tidak")
            special_needs = st.selectbox("Kebutuhan Khusus Pendidikan", [1, 0], format_func=lambda x: "Ya" if x == 1 else "Tidak")
            debtor = st.selectbox("Berhutang", [1, 0], format_func=lambda x: "Ya" if x == 1 else "Tidak")
            tuition_fees = st.selectbox("Biaya Kuliah Lunas", [1, 0], format_func=lambda x: "Ya" if x == 1 else "Tidak")
            gender = st.selectbox("Jenis Kelamin", [1, 0], format_func=lambda x: "Pria" if x == 1 else "Wanita")
            scholarship = st.selectbox("Pemegang Beasiswa", [1, 0], format_func=lambda x: "Ya" if x == 1 else "Tidak")
            age = st.number_input("Usia saat Pendaftaran", min_value=17, max_value=70, value=20)
            international = st.selectbox("Internasional", [1, 0], format_func=lambda x: "Ya" if x == 1 else "Tidak")
            units_1st_credited = st.number_input("Unit Kredit Semester 1", min_value=0, max_value=20, value=0)
            units_1st_enrolled = st.number_input("Unit Terdaftar Semester 1", min_value=0, max_value=20, value=0)
            units_1st_evaluations = st.number_input("Evaluasi Semester 1", min_value=0, max_value=20, value=0)

        with col3:
            units_1st_approved = st.number_input("Unit Disetujui Semester 1", min_value=0, max_value=20, value=0)
            grade_1st = st.number_input("Nilai Semester 1", min_value=0.0, max_value=20.0, value=0.0)
            units_1st_without_evals = st.number_input("Unit Tanpa Evaluasi Semester 1", min_value=0, max_value=20, value=0)
            units_2nd_credited = st.number_input("Unit Kredit Semester 2", min_value=0, max_value=20, value=0)
            units_2nd_enrolled = st.number_input("Unit Terdaftar Semester 2", min_value=0, max_value=20, value=0)
            units_2nd_evaluations = st.number_input("Evaluasi Semester 2", min_value=0, max_value=20, value=0)
            units_2nd_approved = st.number_input("Unit Disetujui Semester 2", min_value=0, max_value=20, value=0)
            grade_2nd = st.number_input("Nilai Semester 2", min_value=0.0, max_value=20.0, value=0.0)
            units_2nd_without_evals = st.number_input("Unit Tanpa Evaluasi Semester 2", min_value=0, max_value=20, value=0)
            unemployment_rate = st.number_input("Tingkat Pengangguran", min_value=0.0, max_value=20.0, value=10.0)
            inflation_rate = st.number_input("Tingkat Inflasi", min_value=-2.0, max_value=5.0, value=2.0)
            gdp = st.number_input("Produk Domestik Bruto", min_value=-5.0, max_value=5.0, value=0.0)

        submitted = st.form_submit_button("Prediksi")

        if submitted:
            # Buat DataFrame untuk input dengan 36 fitur
            input_data = pd.DataFrame({
                'Marital_status': [marital_status],
                'Application_mode': [application_mode],
                'Application_order': [application_order],
                'Course': [course],
                'Daytime_evening_attendance': [daytime_attendance],
                'Previous_qualification': [previous_qualification],
                'Previous_qualification_grade': [previous_qualification_grade],
                'Nacionality': [nationality],
                "Mother's_qualification": [mothers_qualification],
                "Father's_qualification": [fathers_qualification],
                "Mother's_occupation": [mothers_occupation],
                "Father's_occupation": [fathers_occupation],
                'Admission_grade': [admission_grade],
                'Displaced': [displaced],
                'Educational_special_needs': [special_needs],
                'Debtor': [debtor],
                'Tuition_fees_up_to_date': [tuition_fees],
                'Gender': [gender],
                'Scholarship_holder': [scholarship],
                'Age_at_enrollment': [age],
                'International': [international],
                'Curricular_units_1st_sem_credited': [units_1st_credited],
                'Curricular_units_1st_sem_enrolled': [units_1st_enrolled],
                'Curricular_units_1st_sem_evaluations': [units_1st_evaluations],
                'Curricular_units_1st_sem_approved': [units_1st_approved],
                'Curricular_units_1st_sem_grade': [grade_1st],
                'Curricular_units_1st_sem_without_evaluations': [units_1st_without_evals],
                'Curricular_units_2nd_sem_credited': [units_2nd_credited],
                'Curricular_units_2nd_sem_enrolled': [units_2nd_enrolled],
                'Curricular_units_2nd_sem_evaluations': [units_2nd_evaluations],
                'Curricular_units_2nd_sem_approved': [units_2nd_approved],
                'Curricular_units_2nd_sem_grade': [grade_2nd],
                'Curricular_units_2nd_sem_without_evaluations': [units_2nd_without_evals],
                'Unemployment_rate': [unemployment_rate],
                'Inflation_rate': [inflation_rate],
                'GDP': [gdp]
            })

            # Pastikan urutan kolom sesuai dengan model
            expected_features = ['Marital_status', 'Application_mode', 'Application_order', 'Course', 
                                'Daytime_evening_attendance', 'Previous_qualification', 
                                'Previous_qualification_grade', 'Nacionality', "Mother's_qualification", 
                                "Father's_qualification", "Mother's_occupation", "Father's_occupation", 
                                'Admission_grade', 'Displaced', 'Educational_special_needs', 'Debtor', 
                                'Tuition_fees_up_to_date', 'Gender', 'Scholarship_holder', 
                                'Age_at_enrollment', 'International', 'Curricular_units_1st_sem_credited', 
                                'Curricular_units_1st_sem_enrolled', 'Curricular_units_1st_sem_evaluations', 
                                'Curricular_units_1st_sem_approved', 'Curricular_units_1st_sem_grade', 
                                'Curricular_units_1st_sem_without_evaluations', 'Curricular_units_2nd_sem_credited', 
                                'Curricular_units_2nd_sem_enrolled', 'Curricular_units_2nd_sem_evaluations', 
                                'Curricular_units_2nd_sem_approved', 'Curricular_units_2nd_sem_grade', 
                                'Curricular_units_2nd_sem_without_evaluations', 'Unemployment_rate', 
                                'Inflation_rate', 'GDP']
            input_data = input_data[expected_features]

            # Prediksi
            try:
                prediction = model.predict(input_data)
                predicted_status = le.inverse_transform(prediction)[0]
                st.success(f"Prediksi Status: **{predicted_status}**")
                if predicted_status == "Dropout":
                    st.warning("Siswa ini berisiko dropout. Pertimbangkan bimbingan khusus.")
            except Exception as e:
                st.error(f"Gagal melakukan prediksi: {str(e)}")
# Footer
st.markdown("---")
st.markdown("Dibuat oleh Zid Irsyadin Sartono Wijaogy untuk Proyek Akhir Jaya Jaya Institut.")