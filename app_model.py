import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import LocalOutlierFactor
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# from collections import Counter
import math
import os
from sklearn.metrics import accuracy_score,f1_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# configurasi tata letak strm
apptitle = 'Cek Gizi'
st.set_page_config(page_title=apptitle, page_icon=":stethoscope:")

dataset_puskesmas='https://drive.google.com/uc?id=1_a6uKLJrEqF90wmkNuTxgt2wTBCbKmkv'
dataset_kaggle= 'https://drive.google.com/uc?id=1x0qhULl__-JKq8_n4-nkF7qtH-8hIRix'
model= r"model_gizi.sav"
label_map = {
        0: "Gizi Baik",
        1: "Gizi Kurang",
        2: "Gizi Lebih"
    }
# membaca model 
try:
    model_gizi = pickle.load(open(model, 'rb'))
except FileNotFoundError:
    st.error("File model tidak ditemukan. Pastikan path file benar.")
# Ambil semua dokumentasi preprocessing dari pickle
df_imputasi = model_gizi.get('dok_misval')
df_encode = model_gizi.get('dok_encoded')
df_scaling = model_gizi.get('dok_scaled')
df_conv = model_gizi.get('dok_usia')

# Ambil data split
split_data = model_gizi.get('split')
if split_data:
    X_train = split_data['X_train']
    X_test = split_data['X_test']
    y_train = split_data['y_train']
    y_test = split_data['y_test']

# Ambil hasil SMOTE-ENN
smoteenn_data = model_gizi.get('smoteenn')
if smoteenn_data:
    X_res = smoteenn_data['X_res']
    y_res = smoteenn_data['y_res']
# load model preprocesing
encoders = model_gizi.get('encoder')      # label encoder
scaler = model_gizi.get('scaler')         # scaler MinMax/Standard
model_adb = model_gizi['adaboost']['models']
betas_adb = model_gizi['adaboost']['betas']
clas_adb = np.array(model_gizi['adaboost']['classes'])
fitur_asli = model_gizi['input_features']
outlier =model_gizi.get('dok_outlier')  


# Membaca dataset jika tersedia
df_puskesmas = None
df_kaggle = None

kolom_kategorik=['JK']
target='Status Gizi'
kolom_numerik=['Usia', 'Berat', 'Tinggi', 'LiLA']
fitur=['Usia', 'Berat', 'Tinggi', 'LiLA','JK']
data_latih= model_gizi['data_latih']

try:
    df_puskesmas = pd.read_csv(dataset_puskesmas)
    df_kaggle = pd.read_csv(dataset_kaggle)
except Exception as e:
    st.error(f"❌ Gagal memuat dataset: {e}")


st.image("judul.png", use_container_width=True )
st.markdown(
        """
        klasifikasi merupkan teknik data mining dengan mengelompokkan data berdasarkan karakteristik tertentu.
        metode untuk melakukan klasifikasi beragam jenisnya sala satunya yang digunakan pada penelitian ini yaitu Algoritma C5.0 dengan metode
        penguatan AdaBoost yang cara kerjanya membuat pohon keputusan setiap iterasi (n_estimator) dengan bobot berdasarkan eror pada iterasi 
        sebelumnya.

        Total data yang digunakan sebesar 2.759 data status gizi pada balita yang berasal dari dua sumber yaitu :
        - UPT Puskesmas Sembayat berjumlah 1.477 balita
        - Kaggle berjumlah 1.289 balita 
        
        Data yang digunakan terdapat 6 atribut diantaranya jenis kelamin, usia saat ini, Berat, Tinggi dan LILA (Lingkar Lengan Atas) dan status gizi
        
        Kelas status gizi yang akan diklasifikasikan terdiri dari 3 kelas diantaranya gizi baik, Gizi Rendah dan gizi berlebih
        
        """
    )
tab1, tab2, tab3 = st.tabs(["Informasi data", "Preprocesing", "Performa model"])
if df_puskesmas is not None and df_kaggle is not None:
    with tab1:
            st.success("Dataset berhasil dimuat!")
            st.markdown("## sampel Dataset status gizi UPT puskesmas ")
            st.dataframe(df_puskesmas.head())
            st.markdown("## sampel Dataset status gizi Kaggle ")
            st.dataframe(df_kaggle.head())
            st.markdown("## Distribusi data ")
            df_puskesmas.rename(columns={'Usia Saat Ukur': 'Usia'}, inplace=True)
            target='Status Gizi'
            fitur=['JK','Usia','Berat','Tinggi','LiLA']
            x_kaggle=df_kaggle[fitur]
            x_puskesmas=df_puskesmas[fitur]
            y_kaggle=df_kaggle[target]
            y_puskesmas=df_puskesmas[target]
            x_gizi = pd.concat([x_puskesmas,x_kaggle], axis=0, ignore_index=True)
            y_gizi = pd.concat([y_puskesmas,y_kaggle], axis=0, ignore_index=True)
            gizi_df = x_gizi.copy()
            gizi_df['Status Gizi'] = y_gizi.values
            mapping_gizi = {
                'Gizi Buruk': 'Gizi Kurang',
                'Gizi Kurang': 'Gizi Kurang',
                'Gizi Lebih': 'Gizi Lebih',
                'Resiko Gizi Lebih': 'Gizi Lebih',
                'Risiko Gizi Lebih': 'Gizi Lebih',
                'Obesitas': 'Gizi Lebih',
                'Gizi Baik': 'Gizi Baik'
            }
            gizi_df['Status Gizi'] = gizi_df['Status Gizi'].map(mapping_gizi)
            gizi_df.head()
            st.markdown("## Distribusi data berdasarkan kelas status gizi")
            class_counts = gizi_df['Status Gizi'].value_counts()
            try:
                fig, ax = plt.subplots(figsize=(6, 4))
                
                sns.barplot(
                    x=class_counts.index,
                    y=class_counts.values,
                    palette="rocket",  # ✅ palette bisa dipakai di sini
                    ax=ax
                )

                # Tambahkan nilai di atas batang
                for p in ax.patches:
                    yval = p.get_height()
                    ax.text(
                        p.get_x() + p.get_width() / 2,
                        yval + 8,
                        str(int(yval)),
                        ha='center',
                        va='bottom',
                        fontsize=10,
                        fontweight='bold'
                    )

                ax.set_title("Distribusi Status Gizi")
                ax.set_xlabel("Kategori Status Gizi")
                ax.set_ylabel("Jumlah Data")
                st.pyplot(fig, clear_figure=True)

                with st.expander("See notes"):
                    st.markdown("""
                    * Berdasarkan grafik tersebut jumlah data pada setiap kelasnya memiliki ketimpangan dengan rasio 61%:26%:13%
                    * Ketimpangan kelas tersebut, pada penelitian ini diatasi dengan metode Hybrid sampling (SMOTE-ENN) yang cara kerjanya yaitu :
                    1. menambahkan jumlah data pada kelas minoritas (Gizi Kurang dan Gizi Lebih) hingga sama dengan jumlah pada kelas mayoritas (Gizi Baik) => SMOTE. 
                    2. lalu, semua data dari kelas mayoritas maupun minoritas di evaluasi berdasarkan tetangga terdekatnya yang dihitung menggunakan persamaan euclidean. 
                    3. Cek label mayoritas tetangga terdekatnya masing masing data apakah sama
                                dengan label aslinya jika tidak sama data tersebut dianggap noise dan dihapus.
                    """)
            except Exception as e:
                st.error(f"❌ Gagal membuat grafik: {e}")

            st.markdown("## Distrbusi data fitur Jenis kelamin berdasarkan kelas status gizi")
            # Bersihkan spasi dulu
            try:
                gizi_df['JK'] = gizi_df['JK'].astype(str).str.strip()
                gizi_df['Status Gizi'] = gizi_df['Status Gizi'].astype(str).str.strip()
                fig, ax = plt.subplots(figsize=(8, 5))
                sns.countplot(data=gizi_df, x='JK', hue='Status Gizi', palette="rocket", ax=ax)
                plt.title('Hubungan Jenis Kelamin (JK) dengan Status Gizi')
                plt.xlabel('Jenis Kelamin')
                plt.ylabel('Jumlah')
                plt.grid(axis='y', linestyle='--', alpha=0.5)

                # Label jumlah di atas bar
                for p in ax.patches:
                    height = p.get_height()
                    if height > 0:  # supaya tidak error kalau kosong
                        ax.text(
                            p.get_x() + p.get_width()/2,
                            height + 0.3,
                            int(height),
                            ha='center',
                            fontsize=10
                        )
                st.pyplot(fig, clear_figure=True)
                with st.expander("See notes"):
                    st.markdown("""
                    * Berdasarkan grafik distribusi diatas balita dengan jenis kelamin laki-laki mendominasi di setiap kelas status gizi 
                                dibandingkan pada balita dengan jenis perempuan """)
            except Exception as e:
                st.error(f"❌ Gagal membuat grafik: {e}")

            st.markdown("## Distribusi data fitur Berat badan berdasarkan kelas status gizi")
            try:
                kelas_gizi = gizi_df['Status Gizi'].unique()
                n_kelas = len(kelas_gizi)

                fig, axes = plt.subplots(nrows=n_kelas, ncols=2, figsize=(14, 5 * n_kelas))

                for i, kelas in enumerate(kelas_gizi):
                    data_kelas = gizi_df[gizi_df['Status Gizi'] == kelas]

                    # Histogram per kelas
                    sns.histplot(data_kelas["Berat"], kde=True, ax=axes[i, 0], color=sns.color_palette("rocket", n_kelas)[i])
                    axes[i, 0].set_title(f'Histogram Berat - {kelas} (KDE)', fontsize=12)
                    axes[i, 0].set_xlabel("Berat")
                    axes[i, 0].set_ylabel('Frekuensi')

                    # Boxplot per kelas
                    sns.boxplot(x=data_kelas["Berat"], ax=axes[i, 1], color=sns.color_palette("rocket", n_kelas)[i])
                    axes[i, 1].set_title(f'Box Plot Berat - {kelas} (Deteksi Outlier)', fontsize=12)
                    axes[i, 1].set_xlabel("Berat")

                plt.suptitle("Distribusi Berat Berdasarkan Kelas Status Gizi", fontsize=14, fontweight='bold', y=1.01)
                plt.tight_layout()
                st.pyplot(fig, clear_figure=True)

                with st.expander("See notes"):
                    st.markdown("""
                    * Berdasarkan grafik histogram fitur berat, disetiap kelasnya yaitu :
                    1. Pada gizi baik sebaranya merata dengan puncak di sekitar 10-13 kg dengan bentuk distribusinya bentuk lonceng (normal)
                    2. Pada gizi kurang distrbusinya memiliki dua puncak yaitu di sekitar 9 kg dan 13-14 kg hal ini memungkinkan adanya karakteristik kedua dataset yang berbeda dalam penentuan gizi kurang 
                    dan bisa memungkinkan disebabkan adanya kelompok usia yang berbeda. namun penyebaran berat pada gizi kurang dari 2-16 kg.
                    3. pada gizi lebih memiliki distrbusi miring ke kanan yang mengindikasikan sebaran beratnya lebih berat dibandingkan kelas lain dengan titik puncak sekitar 11-12 kg dan ekor panjang hingga 30 kg 
                    * Berdasarkan grafik blox plot terlihat adanya data pencilan/outlier tiap kelasnya :
                    1. Pada gizi baik, memiliki nilai median sekitar 11-12 kg dengan nilai IQR 9-13 kg yang mengindikasikan data cukup terpusat dan terdapat outlier di batas bawa extreme yaitu sekitar 2,5 kg dan di  batas atas extrem sekitar 19-20 kg
                    2. Pada gizi kurang, memiliki nilai median sekitar 10 kg dengan nilai IQR lebih lebar sekiat 8-13 kg dan terdapat outlier di batas bawa extreme yaitu sekitar 2 kg yang mengindikasikan ada balita dengan berat badan lebih ringan yang jauh dibawah rata-rata kelasnya
                    3. pada gizi lebih,memiliki nilai median sekitar 13-14 kg dengan nilai IQR lebi tinggi 10-16 kg dan terdapat outlier di batas atas extreme yaitu sekitar 25-30 kg yang mengindikasikan ada balita dengan berat badan yang sangat berat/ diatas rata rata dikelasnya
                    * Analisis distribusi dilakukan secara terpisah per kelas Status Gizi (Gizi Baik, Gizi Kurang, 
                    dan Gizi Lebih) karena apabila seluruh data digabung dalam satu grafik, distribusi yang 
                    dihasilkan akan bersifat umum dan menyembunyikan pola tersembunyi di masing-masing kelas. 
                    """)

            except Exception as e:
                st.error(f"❌ Gagal membuat grafik: {e}")

            st.markdown("## Distribusi data fitur tinggi badan berdasarkan kelas status gizi")
            try:
                kelas_gizi = gizi_df['Status Gizi'].unique()
                n_kelas = len(kelas_gizi)

                fig, axes = plt.subplots(nrows=n_kelas, ncols=2, figsize=(14, 5 * n_kelas))

                for i, kelas in enumerate(kelas_gizi):
                    data_kelas = gizi_df[gizi_df['Status Gizi'] == kelas]

                    # Histogram per kelas
                    sns.histplot(data_kelas["Tinggi"], kde=True, ax=axes[i, 0], color=sns.color_palette("rocket", n_kelas)[i])
                    axes[i, 0].set_title(f'Histogram Tinggi - {kelas} (KDE)', fontsize=12)
                    axes[i, 0].set_xlabel("Tinggi")
                    axes[i, 0].set_ylabel('Frekuensi')

                    # Boxplot per kelas
                    sns.boxplot(x=data_kelas["Tinggi"], ax=axes[i, 1], color=sns.color_palette("rocket", n_kelas)[i])
                    axes[i, 1].set_title(f'Box Plot Tinggi - {kelas} (Deteksi Outlier)', fontsize=12)
                    axes[i, 1].set_xlabel("Tinggi")

                plt.suptitle("Distribusi Tinggi Berdasarkan Kelas Status Gizi", fontsize=14, fontweight='bold', y=1.01)
                plt.tight_layout()
                st.pyplot(fig, clear_figure=True)

                with st.expander("See notes"):
                    st.markdown("""
                        * Berdasarkan grafik histogram fitur Tinggi, distribusi pada setiap kelas status gizi dapat dijelaskan sebagai berikut:
                        
                        1. **Gizi Baik** — Data cenderung berdistribusi **mendekati normal** dengan puncak (modus) berada pada rentang tinggi 85–95, namun masih terdapat sedikit kemencengan (skewness) ke kiri akibat adanya nilai tinggi yang rendah (di bawah 55).
                        2. **Gizi Kurang** — Data menunjukkan pola distribusi **bimodal** (dua puncak), yaitu pada rentang 85–90 dan 100–105, yang mengindikasikan adanya dua sub-kelompok data dengan karakteristik tinggi yang berbeda.
                        3. **Gizi Lebih** — Data terkonsentrasi pada rentang tinggi 65–75 dengan bentuk distribusi yang **menceng ke kanan (right-skewed)**, di mana terdapat ekor panjang pada nilai tinggi yang lebih besar (di atas 100).
                        
                        * Sementara itu, berdasarkan grafik box plot, terlihat adanya data pencilan (*outlier*) pada setiap kelasnya:
                        
                        1. **Gizi Baik** — Teridentifikasi beberapa titik outlier pada sisi bawah (nilai tinggi rendah, berkisar di bawah 50), yang berada di luar batas *whisker* bawah.
                        2. **Gizi Kurang** — Teridentifikasi beberapa titik outlier pada sisi bawah (nilai tinggi rendah, berkisar di bawah 55), dengan jumlah outlier yang relatif lebih banyak dibandingkan kelas lainnya.
                        3. **Gizi Lebih** — Tidak teridentifikasi adanya titik outlier, baik pada sisi atas maupun bawah, yang mengindikasikan bahwa seluruh data pada kelas ini masih berada dalam rentang normal (*whisker* atas dan bawah).
                        
                    * Analisis distribusi dilakukan secara terpisah per kelas Status Gizi (Gizi Baik, Gizi Kurang, 
                    dan Gizi Lebih) karena apabila seluruh data digabung dalam satu grafik, distribusi yang 
                    dihasilkan akan bersifat umum dan menyembunyikan pola tersembunyi di masing-masing kelas. 
                    * Secara umum, pada nilai tinggi badan ini outlier yang muncul pada kelas Gizi Baik dan Gizi Kurang berada pada sisi bawah (nilai tinggi yang rendah), sehingga perlu dipertimbangkan penanganannya (baik dihapus maupun diimputasi) sebelum data digunakan pada tahap pemodelan, guna menghindari bias hasil klasifikasi.
                        kondisi status gizi.
                    """)

            except Exception as e:
                st.error(f"❌ Gagal membuat grafik: {e}")

            st.markdown("## Distribusi data fitur usia badan berdasarkan kelas status gizi")
            try:
                kelas_gizi = gizi_df['Status Gizi'].unique()
                n_kelas = len(kelas_gizi)

                fig, axes = plt.subplots(nrows=n_kelas, ncols=2, figsize=(14, 5 * n_kelas))

                for i, kelas in enumerate(kelas_gizi):
                    data_kelas = gizi_df[gizi_df['Status Gizi'] == kelas]

                    # Histogram per kelas
                    sns.histplot(data_kelas["Usia"], kde=True, ax=axes[i, 0], color=sns.color_palette("rocket", n_kelas)[i])
                    axes[i, 0].set_title(f'Histogram Usia - {kelas} (KDE)', fontsize=12)
                    axes[i, 0].set_xlabel("Usia")
                    axes[i, 0].set_ylabel('Frekuensi')

                    # Boxplot per kelas
                    sns.boxplot(x=data_kelas["Usia"], ax=axes[i, 1], color=sns.color_palette("rocket", n_kelas)[i])
                    axes[i, 1].set_title(f'Box Plot Usia - {kelas} (Deteksi Outlier)', fontsize=12)
                    axes[i, 1].set_xlabel("Usia")

                plt.suptitle("Distribusi Usia Berdasarkan Kelas Status Gizi", fontsize=14, fontweight='bold', y=1.01)
                plt.tight_layout()
                st.pyplot(fig, clear_figure=True)

                with st.expander("See notes"):
                    st.markdown("""
                    * Berdasarkan grafik histogram fitur Usia, distribusi pada setiap kelas status gizi dapat dijelaskan sebagai berikut:
                        
                        1. **Gizi Baik** — Data sangat terkonsentrasi (menumpuk) pada nilai usia tertinggi, dengan frekuensi yang jauh lebih besar dibanding nilai usia lainnya, sehingga bentuk distribusi **sangat menceng ke kiri (left-skewed)** dan hampir menyerupai nilai konstan pada rentang usia tersebut.
                        2. **Gizi Kurang** — Data juga terkonsentrasi pada rentang usia tertinggi dengan pola distribusi yang **menceng tajam ke kiri (left-skewed)**, namun terdapat sedikit sebaran nilai pada rentang usia yang lebih rendah dibanding kelas lainnya.
                        3. **Gizi Lebih** — Pola distribusi serupa dengan kedua kelas sebelumnya, yaitu data terpusat pada nilai usia tertinggi dengan bentuk **menceng ke kiri (left-skewed)**, meskipun sebaran pada rentang usia rendah relatif lebih banyak dibanding kelas Gizi Baik.
                        
                        * Sementara itu, berdasarkan grafik box plot, terlihat adanya data pencilan (*outlier*) pada setiap kelasnya:
                        
                        1. **Gizi Baik** — Tidak teridentifikasi adanya titik outlier, baik pada sisi atas maupun bawah, yang menunjukkan bahwa seluruh data usia pada kelas ini masih berada dalam rentang normal (*whisker* atas dan bawah).
                        2. **Gizi Kurang** — Teridentifikasi **outlier dalam jumlah sangat banyak** pada sisi bawah (nilai usia yang rendah), yang berada jauh di luar batas *whisker* bawah, mengindikasikan sebaran data usia yang tidak merata pada kelas ini.
                        3. **Gizi Lebih** — Tidak teridentifikasi adanya titik outlier, baik pada sisi atas maupun bawah, sehingga data usia pada kelas ini juga relatif seragam dan berada dalam rentang normal.
                        
                        * Secara umum, outlier yang signifikan hanya ditemukan pada kelas **Gizi Kurang**, sehingga perlu dipertimbangkan penanganannya (baik dihapus maupun diimputasi) sebelum data digunakan pada tahap pemodelan, guna menghindari bias hasil klasifikasi.
                    * Analisis distribusi dilakukan secara terpisah per kelas Status Gizi (Gizi Baik, Gizi Kurang, 
                    dan Gizi Lebih) karena apabila seluruh data digabung dalam satu grafik, distribusi yang 
                    dihasilkan akan bersifat umum dan menyembunyikan pola tersembunyi di masing-masing kelas. 
                    
                    """)

            except Exception as e:
                st.error(f"❌ Gagal membuat grafik: {e}")

            st.markdown("## Distribusi data fitur LiLA badan berdasarkan kelas status gizi")
            try:
                kelas_gizi = gizi_df['Status Gizi'].unique()
                n_kelas = len(kelas_gizi)

                fig, axes = plt.subplots(nrows=n_kelas, ncols=2, figsize=(14, 5 * n_kelas))

                for i, kelas in enumerate(kelas_gizi):
                    data_kelas = gizi_df[gizi_df['Status Gizi'] == kelas]

                    # Histogram per kelas
                    sns.histplot(data_kelas["LiLA"], kde=True, ax=axes[i, 0], color=sns.color_palette("rocket", n_kelas)[i])
                    axes[i, 0].set_title(f'Histogram LiLA - {kelas} (KDE)', fontsize=12)
                    axes[i, 0].set_xlabel("LiLA")
                    axes[i, 0].set_ylabel('Frekuensi')

                    # Boxplot per kelas
                    sns.boxplot(x=data_kelas["LiLA"], ax=axes[i, 1], color=sns.color_palette("rocket", n_kelas)[i])
                    axes[i, 1].set_title(f'Box Plot LiLA - {kelas} (Deteksi Outlier)', fontsize=12)
                    axes[i, 1].set_xlabel("LiLA")

                plt.suptitle("Distribusi LiLA Berdasarkan Kelas Status Gizi", fontsize=14, fontweight='bold', y=1.01)
                plt.tight_layout()
                st.pyplot(fig, clear_figure=True)

                with st.expander("See notes"):
                    st.markdown("""
                    * Berdasarkan grafik histogram fitur LiLA, distribusi pada setiap kelas status gizi dapat dijelaskan sebagai berikut:

                        1. **Gizi Baik** — Data terkonsentrasi pada rentang 11–19 dengan puncak (modus) di sekitar nilai 14–15, namun terdapat kelompok data terpisah pada nilai mendekati 0 serta beberapa nilai ekstrem pada rentang 45–50, sehingga bentuk distribusi cenderung **multimodal** dengan ekor panjang ke kanan.
                        2. **Gizi Kurang** — Data juga menunjukkan pola **multimodal**, dengan puncak tertinggi pada rentang nilai mendekati 0, puncak kedua pada rentang 10–16, serta sedikit sebaran pada nilai yang lebih tinggi hingga mendekati 45.
                        3. **Gizi Lebih** — Data terkonsentrasi pada rentang nilai rendah (0–3) sebagai puncak utama, dengan puncak kedua yang lebih kecil pada rentang 14–18, membentuk pola **bimodal** dengan distribusi yang **menceng ke kanan (right-skewed)**.
                * Sementara itu, berdasarkan grafik box plot, terlihat adanya data pencilan (*outlier*) pada setiap kelasnya:

                        1. **Gizi Baik** — Teridentifikasi **outlier dalam jumlah banyak**, baik pada sisi bawah (nilai mendekati 0) maupun sisi atas (nilai di atas 20 hingga mendekati 50), menunjukkan sebaran data yang cukup ekstrem pada kelas ini.
                        2. **Gizi Kurang** — Teridentifikasi beberapa titik outlier pada sisi atas (nilai di atas 40), sementara pada sisi bawah tidak ditemukan outlier karena nilai minimum masih berada dalam batas *whisker*.
                        3. **Gizi Lebih** — Tidak teridentifikasi adanya titik outlier, baik pada sisi atas maupun bawah, yang menunjukkan bahwa seluruh data LiLA pada kelas ini masih berada dalam rentang normal (*whisker* atas dan bawah).

                * Secara umum, outlier paling banyak ditemukan pada kelas **Gizi Baik**, diikuti oleh kelas **Gizi Kurang**, sehingga perlu dipertimbangkan penanganannya (baik dihapus maupun diimputasi) sebelum data digunakan pada tahap pemodelan, guna menghindari bias hasil klasifikasi.
                    * Analisis distribusi dilakukan secara terpisah per kelas Status Gizi (Gizi Baik, Gizi Kurang, 
                    dan Gizi Lebih) karena apabila seluruh data digabung dalam satu grafik, distribusi yang 
                    dihasilkan akan bersifat umum dan menyembunyikan pola tersembunyi di masing-masing kelas. 
                    """)

            except Exception as e:
                st.error(f"❌ Gagal membuat grafik: {e}")

    with tab2 :
            st.markdown("## Preprocesing")
            st.markdown("### 1. Imputasi Missing Value dengan metode mean")
            st.markdown(""" Cek missing value tiap kolom """)
            st.dataframe(gizi_df.isna().sum())
            st.markdown(""" setelah imputasi missing value """)
            st.write(df_imputasi.isna().sum())
            st.dataframe(df_imputasi.head())


            st.markdown("### 2. Transformasi Data")
            st.markdown(""" pada proses ini fitur usia akan ditransformasi guna seragam dan mudah diproses oleh program. 
                            jadi dari format Hari/Bulan/Tahun menjadi Bulan """)
            st.dataframe(df_conv.head())
                
            st.markdown("### 3. Normalisasi")
            st.markdown(""" data dinormalisasi agar nilainya dalam rentang 0-1 menggunakan metode
                            normalisasi min max. tujuan proses ini agar jarak yang dihasilkan seragam. 
                            diterapkan pada fitur numerik""")
            st.dataframe(df_scaling.head())

            st.markdown("### 4. Label Encoding")
            st.markdown(""" pada proses ini fitur Status Gizi dan Jenis Kelamin akan diubah kedalam format numerik sesuai dengan ketentuan berikut """)

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Jenis Kelamin**")
                jk_df = pd.DataFrame({
                    "Jenis Kelamin": ["L", "P"],
                    "Label": [0, 1]
                })
                st.dataframe(jk_df, hide_index=True, use_container_width=True)

            with col2:
                st.markdown("**Status Gizi**")
                gizi_df_encoding = pd.DataFrame({
                    "Status Gizi": ["Gizi Baik", "Gizi Kurang", "Gizi Lebih"],
                    "Label": [0, 1, 2]
                })
                st.dataframe(gizi_df_encoding, hide_index=True, use_container_width=True)
            st.dataframe(df_encode.head())

            st.markdown("### 5. Outlier")
            st.markdown(""" pada proses ini akan dilakukan pengecekan outlier menggunakan metode Local Outlier Factor (LOF)
                            dengan nilai k default(20) """)
            st.dataframe(outlier.head())
            gizi_df= outlier[outlier['LOF_label'] == 1].copy()
            gizi_df.drop(columns=['LOF_label', 'LOF_score'], inplace=True)

            st.markdown("### 6. Pembagian Data")
            st.markdown(""" Data kemudian dibagi menjadi dua bagian dengan rasio 90:10, 
                            yaitu 90% untuk data latih dan 10% untuk data uji.""")
            split_data = model_gizi['split']
            st.write("Jumlah data total :", len(gizi_df))
            st.write("Jumlah data training :", len(X_train))
            st.write("Jumlah data testing :", len(X_test))
                

            st.markdown("## SMOTE-ENN")
            st.markdown(""" Dikarenakan jumlah tiap kelas tidak seimbang. maka, disini 
                            SMOTE-ENN digunakan untuk mesintesis data pada kelas minoritas dan menghapus data 
                            yang tidak represemtatif agar jumlahnya seimbang dam menghasilkan model yang optimal""")
            st.markdown("### Jumlah tiap kelas sebelum SMOTE-ENN")
            before_counts = y_train.value_counts()
            fig_before, ax_before = plt.subplots(figsize=(6,4))
            bars = ax_before.bar(before_counts.index, before_counts.values)

            for bar in bars:
                yval = bar.get_height()
                ax_before.text(bar.get_x() + bar.get_width()/2, yval + 5, str(yval),
                                ha='center', fontsize=10, fontweight="bold")
            ax_before.set_title("Distribusi Kelas Sebelum SMOTE=ENN")
            ax_before.set_xlabel("Kelas")
            ax_before.set_ylabel("Jumlah")
            st.pyplot(fig_before)

            st.markdown("### Proses SMOTE-ENN")
            st.markdown("""
                1. Masukkan data training status gizi yang tidak seimbang.
                2. Pilih data secara acak dari kelas minoritas.
                3. Identifikasi k tetangga terdekat dari kelas minoritas dan pilih satu tetangga secara acak. Proses pencarian tetangga terdekat pada penelitian ini menggunakan perhitungan jarak Euclidean.
                4. Tambahkan data sintesis baru dengan menggunakan rumus SMOTE.
                5. Lakukan berulang dari poin 2 sampai 4 sampai data dari kelas minoritas seimbang dengan kelas mayoritas.
                6. Setelah dataset masing-masing kelas seimbang, masuk ke tahap proses ENN dengan mengidentifikasi k-tetangga terdekat dari setiap data dalam dataset menggunakan persamaan rumus Euclidean distance sebagaimana dirumuskan pada persamaan 2.3.
                7. Cek label mayoritas dari tetangga terdekat masing-masing data, apakah sama dengan label aslinya. Jika tidak sama, data tersebut dianggap noise dan dihapus.

                Dengan demikian, hasil dari proses ini menghasilkan dataset training yang seimbang apabila tidak terdapat data noise, atau hampir seimbang apabila masih terdapat data noise yang perlu dibersihkan.
                """)
                            
            # --- Setelah SMOTE-ENN --
            st.markdown("### Jumlah tiap kelas setelah SMOTE-ENN")
            sm = model_gizi['smoteenn']
            after_counts = y_res.value_counts()
            fig_after, ax_after = plt.subplots(figsize=(6,4))
            bars = ax_after.bar(after_counts.index, after_counts.values)

            for bar in bars:
                yval = bar.get_height()
                ax_after.text(bar.get_x() + bar.get_width()/2, yval + 5, str(yval),
                            ha='center', fontsize=10, fontweight="bold")

            ax_after.set_title("Distribusi Kelas Setelah SMOTE-ENN")
            ax_after.set_xlabel("Kelas")
            ax_after.set_ylabel("Jumlah")
            st.pyplot(fig_after)
    with tab3:
            st.header("📊 Hasil model terbaik dari skenario penelitian")
            st.markdown("""
                Penelitian ini telah menguji sebanyak 72 sub-skenario penelitian dengan menguji beberapa parameter dan kondisi, di antaranya:

                1. Penanganan outlier dengan menghapus data yang terindikasi outlier.
                2. Penanganan outlier dengan imputasi data menggunakan metode Winsorization.
                3. Penanganan outlier dengan membiarkan data yang terindikasi outlier tanpa penanganan.
                4. Parameter SMOTE [3, 4, 5, 6, 7].
                5. Parameter ENN [3, 4, 5, 6, 7].
                6. Penanganan data imbalance dengan menggunakan metode SMOTE.
                7. Penanganan data imbalance dengan menggunakan metode ENN.
                8. Penanganan data imbalance dengan menggunakan metode SMOTE-ENN.
                9. Pembagian data training dan testing 70:30.
                10. Pembagian data training dan testing 80:20.
                11. Pembagian data training dan testing 90:10.
                12. Parameter max_depth pada algoritma C5.0 [3, 4, 5, 6, 7].
                13. Parameter n_estimators pada AdaBoost [50, 100].
                14. Metode algoritma C5.0 tunggal.
                15. Metode ensemble C5.0 sebagai weak learner dari metode boosting AdaBoost.

                Dari 72 sub-skenario yang diujikan pada penelitian ini, diperoleh **kombinasi parameter dan kondisi terbaik** berdasarkan hasil tuning Grid Search, yaitu pada pembagian data training dan testkondisi **penghapusan outlier** dengan metode penanganan imbalance **SMOTE-ENN**, menggunakan kombinasi parameter **k_neighbors = 3**, **n_neighbors = 5**, **max_depth = 7**, dan **n_estimators = 50**, dengan metode **ensemble Algoritma C5.0-AdaBoost**.
                """)
                # ============================
                # ======= MODEL C5.0 =========
                # ============================
            st.subheader("Berikut hasil performa dari sub-skenario terbaik")

            tree_c50 = model_gizi['algoritma.C5']['tree']
            depth = model_gizi['algoritma.C5']['depth']

            X_test = model_gizi['X_test']
            y_test = model_gizi['y_test']

            def mayoritas_kelas(gizi, target, default=None):
                if gizi.empty:
                    return default
                return (gizi.groupby(target)['bobot'].sum().idxmax())

            def prediksi_c50(baris, node, data_latih, target):
                if not isinstance(node, dict):
                    return node
                atr = node['atribut']
                if atr in kolom_kategorik:
                    nilai = baris[atr]
                    if nilai in node:
                        return prediksi_c50(baris, node[nilai], data_latih, target)
                    else:
                        return mayoritas_kelas(data_latih, target, default=mayoritas_kelas(data_latih, target))
                else:
                    nilai = baris[atr]
                    batas = node['mean']
                    if nilai < batas:
                        return prediksi_c50(baris, node['kiri'], data_latih, target)
                    else:
                        return prediksi_c50(baris, node['kanan'], data_latih, target)

            # Prediksi semua data test C5.0
            y_pred = [
                prediksi_c50(X_test.iloc[i], tree_c50, data_latih, target)
                for i in range(len(X_test))
            ]

            # Paksa y_test jadi angka
            y_test = pd.Series(y_test).astype(int)

            # Mapping label string → angka
            reverse_label_map = {
                "Gizi Baik": 0,
                "Gizi Kurang": 1,
                "Gizi Lebih": 2
            }
            # Mapping angka → label (untuk tampilan confusion matrix)
            label_map = {v: k for k, v in reverse_label_map.items()}

            # Paksa semua prediksi jadi string dulu lalu map
            y_pred = [str(i).strip() for i in y_pred]
            y_pred = [reverse_label_map.get(i, 0) for i in y_pred]

            # =======================
            #     EVALUASI C5.0
            # =======================
            acc = accuracy_score(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred)

            precision, recall, f1, _ = precision_recall_fscore_support(
                y_test, y_pred, average="macro", zero_division=0
            )

            sensitivity = recall
            gmean = math.sqrt(precision * recall)

            # =======================
            #   TAMPILAN EVALUASI (RAPI, TANPA TABEL)
            # =======================
            st.subheader("📌 Evaluasi Model C5.0")

            col1, col2, col3 = st.columns(3)
            col1.metric("Accuracy", f"{acc:.2%}")
            col2.metric("Precision", f"{precision:.2%}")
            col3.metric("Recall (Sensitivity)", f"{recall:.2%}")

            col4, col5 = st.columns(2)
            col4.metric("F1-Score", f"{f1:.2%}")
            col5.metric("G-Mean", f"{gmean:.2%}")

            # =======================
            #   CONFUSION MATRIX DENGAN LABEL NAMA KELAS
            # =======================
            st.subheader("📌 Confusion Matrix")

            labels_urut = [0, 1, 2]  # urutan kelas sesuai reverse_label_map
            nama_kelas = [label_map[i] for i in labels_urut]

            cm_df = pd.DataFrame(
                cm,
                index=[f"Aktual: {n}" for n in nama_kelas],
                columns=[f"Prediksi: {n}" for n in nama_kelas]
            )

            st.dataframe(cm_df)

                # =======================
                # Sample Prediksi C5.0
                # =======================
            st.subheader("📄 Perbandingan Prediksi vs Aktual")

            sample_size = 10
            sample_df = X_test.head(sample_size).copy()
            sample_df["Actual"] = y_test[:sample_size].values
            sample_df["Predict"] = y_pred[:sample_size]
            st.dataframe(sample_df)


                # ================================================================
                # ====================    ADABOOST + C5.0    ======================
                # ===============================================================

            model_adb = model_gizi['adaboost']['models']
            betas_adb = model_gizi['adaboost']['betas']
            clas_adb = np.array(model_gizi['adaboost']['classes'])   # pastikan array

            # Mapping label kelas -> nama gizi
            label_gizi_map = {
                0: "Gizi Baik",
                1: "Gizi Kurang",
                2: "Gizi Lebih"
            }

            # ============================
            # Fungsi prediksi ADABOOST
            # ============================
            def prediksi_adb(gizi, model, beta):
                hasil = []
                for a in range(len(gizi)):
                    vote_bobot = {}
                    for pohon, b in zip(model, beta):
                        pred = prediksi_c50(gizi.iloc[a], pohon, gizi, target)
                        vote_bobot[pred] = vote_bobot.get(pred, 0) + np.log(1 / b)
                    hasil.append(max(vote_bobot, key=vote_bobot.get))
                return np.array(hasil)

            # Prediksi Adaboost
            y_pred_adb = prediksi_adb(X_test, model_adb, betas_adb)
            y_pred_adb = [reverse_label_map.get(i, i) for i in y_pred_adb]

            # ============================
            #      EVALUASI ADABOOST
            # ============================
            acc_adb = accuracy_score(y_test, y_pred_adb)

            # Urutkan label agar konsisten (0, 1, 2)
            label_order = sorted(label_gizi_map.keys())
            cm_adb = confusion_matrix(y_test, y_pred_adb, labels=label_order)

            precision_adb, recall_adb, f1_adb, _ = precision_recall_fscore_support(
                y_test, y_pred_adb, average="macro", zero_division=0
            )

            gmean_adb = math.sqrt(precision_adb * recall_adb)

            # ============================
            # Tampilan Evaluasi (rapi, tanpa tabel)
            # ============================
            st.subheader("📌 Evaluasi Adaboost (C5.0 Base Learner)")

            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Accuracy", f"{acc_adb:.2%}")
            col2.metric("Precision", f"{precision_adb:.2%}")
            col3.metric("Recall (Sensitivity)", f"{recall_adb:.2%}")
            col4.metric("F1-Score", f"{f1_adb:.2%}")
            col5.metric("G-Mean", f"{gmean_adb:.2%}")

            # ============================
            # Confusion Matrix dengan Label Nama Gizi
            # ============================
            st.subheader("📌 Confusion Matrix Adaboost")

            nama_kelas = [label_gizi_map[i] for i in label_order]
            cm_adb_df = pd.DataFrame(cm_adb, index=nama_kelas, columns=nama_kelas)
            cm_adb_df.index.name = "Aktual"
            cm_adb_df.columns.name = "Prediksi"

            st.dataframe(cm_adb_df)

                # ============================
                # Sample Prediksi Adaboost
                # ============================
            st.subheader("📄 Contoh Perbandingan Prediksi vs Aktual (Adaboost)")

            sample_df_adb = X_test.head(sample_size).copy()
            sample_df_adb["Actual"] = y_test[:sample_size].values
            sample_df_adb["Predict"] = y_pred_adb[:sample_size]

            st.dataframe(sample_df_adb)

else:
    st.warning("Dataset belum dapat dimuat.")

st.sidebar.header("👩‍⚕️ Prediksi Status Gizi Balita")
fitur_asli = model_gizi['split']['X_train'].columns.tolist()

    # === Input Form ===
jk = st.sidebar.selectbox("Jenis Kelamin", encoders['JK'].classes_)
usia = st.sidebar.number_input("Usia Saat Ukur (bulan)", 0.0, 60.0, 12.0)
bb = st.sidebar.number_input("Berat Badan (kg)", 0.0, 50.0, 10.0)
tb = st.sidebar.number_input("Tinggi Badan (cm)", 0.0, 200.0, 50.0)
lila = st.sidebar.number_input("LiLA (cm)", 0.0, 40.0, 10.0)

if st.sidebar.button("Prediksi Status Gizi"):

        # === 1. Encoding JK ===
        jk_encoded = encoders['JK'].transform([jk])[0]

        # === 2. Buat dataframe input ===
        df_input = pd.DataFrame([{
            "JK": jk_encoded,
            "Usia Saat Ukur": usia,
            "Berat": bb,
            "Tinggi": tb,
            "LiLA": lila
        }])
        df_input = df_input.reindex(columns=fitur_asli) 

        # === 3. Scaling sesuai model ===
        kolom_scaling = scaler.feature_names_in_
        df_input_scaled = df_input[kolom_scaling]
        scaled_values = scaler.transform(df_input_scaled)
        df_scaled = pd.DataFrame(scaled_values, columns=kolom_scaling)

        def prediksi_c50(baris, node, data_latih, target):
            if not isinstance(node, dict):
                return node
            atr = node['atribut']
            if atr in kolom_kategorik:
                nilai = baris[atr]
                if nilai in node:
                    return prediksi_c50(baris, node[nilai], data_latih, target)
                else:
                    return mayoritas_kelas(data_latih, target,default=mayoritas_kelas(data_latih, target))
            else:
                nilai = baris[atr]
                batas = node['mean']
                if nilai < batas:
                    return prediksi_c50(baris, node['kiri'], data_latih, target)
                else:
                    return prediksi_c50(baris, node['kanan'], data_latih, target)

        def prediksi_adb(gizi,model,beta):
            hasil=[]
            for a in range(len(gizi)):
                vote_bobot={}
                for pohon,b in zip(model,beta):
                    pred=prediksi_c50(gizi.iloc[a],pohon,gizi,target)
                    vote_bobot[pred]=vote_bobot.get(pred,0)+np.log(1/b)
                hasil.append(max(vote_bobot, key=vote_bobot.get))
            return np.array(hasil)

        # === 4. Prediksi dengan ADABOOST ===
        df_final = pd.DataFrame(columns=fitur_asli)
        df_final.loc[0] = 0.0  # default isi angka 0

        # isi kolom scaling
        for col in kolom_scaling:
            df_final.at[0, col] = df_scaled.at[0, col]

        # isi JK (yang tidak di-scaling)
        if 'JK' in df_final.columns:
            df_final.at[0, 'JK'] = jk_encoded

        # === 4. Prediksi dengan ADABOOST ===
        y_pred = prediksi_adb(df_final, model_adb, betas_adb)


        # === 5. Mapping hasil ke label asli ===
        pred_label = label_map[int(y_pred[0])]

        # === 6. Tampilkan hasil ===
        st.sidebar.success(f"### 🔍 Hasil Prediksi: **{pred_label}**")
