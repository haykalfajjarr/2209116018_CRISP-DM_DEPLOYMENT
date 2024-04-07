import streamlit as st
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

def histplot(df):
    fig, axes = plt.subplots(4, 1, figsize =(10,25))
    sns.histplot(df['age'], kde=True, ax=axes[0])
    axes[0].set_title('Distribusi Umur')
    axes[0].set_xlabel('Umur')
    axes[0].set_ylabel('Frekuensi')

    sns.histplot(df['bmi'], kde=True, ax=axes[1])
    axes[1].set_title('Distribusi BMI')
    axes[1].set_xlabel('BMI')
    axes[1].set_ylabel('Frekuensi')

    sns.histplot(df['expenses'], kde=True, ax=axes[2])
    axes[2].set_title('Distribusi Biaya')
    axes[2].set_xlabel('Biaya')
    axes[2].set_ylabel('Frekuensi')

    # Plot for another feature if available in your DataFrame
    if 'another_feature' in df.columns:
        sns.histplot(df['another_feature'], kde=True, ax=axes[3])
        axes[3].set_title('Distribusi Another Feature')
        axes[3].set_xlabel('Another Feature')
        axes[3].set_ylabel('Frekuensi')
    else:
        axes[3].axis('off') 

    st.pyplot(fig)
    text = """
    Diagram histogram yang ditampilkan di atas menggambarkan distribusi dari beberapa fitur dalam dataset. Setiap histogram menunjukkan sebaran nilai-nilai dari fitur yang bersangkutan, dengan garis kernel density estimation (KDE) untuk memberikan perkiraan kurva kepadatan probabilitas.

    1. **Distribusi Umur**:
    Histogram pertama menggambarkan distribusi umur dari data. Sumbu x menunjukkan rentang umur, sementara sumbu y menunjukkan frekuensi kemunculan umur tersebut dalam dataset.

    2. **Distribusi BMI (Body Mass Index)**:
    Histogram kedua menampilkan distribusi indeks massa tubuh (BMI) dari data. Sumbu x menunjukkan rentang BMI, sementara sumbu y menunjukkan frekuensi kemunculan BMI tersebut dalam dataset.

    3. **Distribusi Biaya (Expenses)**:
    Histogram ketiga menunjukkan distribusi biaya medis dari data. Sumbu x menampilkan rentang biaya, sementara sumbu y menampilkan frekuensi kemunculan biaya tersebut dalam dataset.
    """
    st.markdown(text)

def heatmap(df):
    plt.figure(figsize=(10, 10))
    plt.title('Relation Between Fiture')
    numeric_cols = df.select_dtypes(include=['int64', 'float64'])
    corr = numeric_cols.corr()
    sns.heatmap(corr, annot=True, cmap="YlGnBu")
    st.pyplot(plt)
    text = 'Diagram Heatmap yang ditampilkan di atas menggambarkan korelasi di antara berbagai kolom dalam kumpulan data ini. Nilai yang lebih tinggi menunjukkan hubungan yang lebih kuat antara kolom-kolom tersebut.'
    st.markdown(text)
    
def compositionAndComparison (df):
    df['sex'] = df['sex'].replace({0: 'Female', 1: 'Male'})

    numerical_columns = df.select_dtypes(include=['number'])

    sex_composition = numerical_columns.groupby(df['sex']).mean()

    plt.figure(figsize=(10, 6))
    sns.heatmap(sex_composition.T, annot=True, fmt=".2f", cmap='YlGnBu') 
    plt.title('Komposisi untuk setiap jenis kelamin')
    plt.xlabel('Jenis Kelamin')
    plt.ylabel('Fitur')
    st.pyplot(plt.gcf())
    text = """
    Diagram heatmap yang ditampilkan di atas menggambarkan komposisi rata-rata dari berbagai fitur dalam dataset, dibagi berdasarkan jenis kelamin (pria dan wanita). Setiap baris dalam heatmap mewakili satu fitur, sementara kolomnya mewakili jenis kelamin.

    - **Komposisi untuk Setiap Jenis Kelamin**:
    Heatmap menunjukkan rata-rata nilai fitur-fitur dalam dataset untuk setiap jenis kelamin. Warna yang lebih terang menunjukkan nilai yang lebih tinggi, sementara warna yang lebih gelap menunjukkan nilai yang lebih rendah.

    """
    st.markdown(text)

def Predict():
    text = """
    Fungsi ini memungkinkan pengguna untuk memasukkan data mereka sendiri dan melakukan prediksi terhadap hasil tertentu menggunakan model yang telah dilatih sebelumnya.

    - **Input User**:
    Pengguna diminta untuk memasukkan beberapa fitur seperti usia (Age), indeks massa tubuh (BMI), status merokok (Smoker), biaya (Expenses), serta opsi untuk memilih beberapa atribut lain seperti wilayah geografis (Region) dan kategori indeks massa tubuh (Body Mass Category).

    - **User Data**:
    Setelah pengguna memasukkan data, informasi yang dimasukkan akan ditampilkan dalam bentuk dataframe, memungkinkan pengguna untuk memverifikasi data yang dimasukkan sebelum melakukan prediksi.

    - **Predict Button**:
    Setelah memasukkan data, pengguna dapat menekan tombol "Predict" untuk melakukan prediksi menggunakan model yang telah dilatih sebelumnya. Prediksi ini berdasarkan pada data yang dimasukkan oleh pengguna.

    - **Predicted Sex**:
    Setelah tombol "Predict" ditekan, hasil prediksi akan ditampilkan, dalam hal ini, prediksi jenis kelamin (Male/Female) berdasarkan data yang dimasukkan oleh pengguna.

    """
    st.markdown(text)
    st.header('Below Is The Predict')
    age = st.number_input('Age', min_value=0, max_value=120, step=1)
    bmi = st.number_input('BMI', min_value=0.0, max_value=100.0, step=0.1)
    smoker = st.selectbox('Smoker', ['No', 'Yes'])
    expenses = st.number_input('Expenses', min_value=0.0, max_value=100000.0, step=0.01)
    region_northeast = st.checkbox('Region Northeast')
    region_northwest = st.checkbox('Region Northwest')
    region_southeast = st.checkbox('Region Southeast')
    region_southwest = st.checkbox('Region Southwest')
    bodymass_category_Normal = st.checkbox('Body Mass Category Normal')
    bodymass_category_Obesity = st.checkbox('Body Mass Category Obesity')
    bodymass_category_Overweight = st.checkbox('Body Mass Category Overweight')
    bodymass_category_Thin = st.checkbox('Body Mass Category Thin')

    user_data = pd.DataFrame({
        'age': [age],
        'bmi': [bmi],
        'smoker': [smoker == 'Yes'],
        'expenses': [expenses],
        'region_northeast': [region_northeast],
        'region_northwest': [region_northwest],
        'region_southeast': [region_southeast],
        'region_southwest': [region_southwest],
        'bodymass_category_Normal': [bodymass_category_Normal],
        'bodymass_category_Obesity': [bodymass_category_Obesity],
        'bodymass_category_Overweight': [bodymass_category_Overweight],
        'bodymass_category_Thin': [bodymass_category_Thin]
    })

    st.subheader('User Data:')
    st.write(user_data)

    button = st.button('Predict')
    if button:
        with open('gnb.pkl', 'rb') as file:
            loaded_model = pickle.load(file)

        predicted_sex = loaded_model.predict(user_data)

        predicted_sex = 'Male' if predicted_sex == 1 else 'Female'

        st.write("Predicted Sex:", predicted_sex)
    
def clustering(df):
    klasifikasi(df)
    st.header('Below Is The Clustering')
    age = st.number_input('Age', min_value=0, max_value=120, step=1)
    bmi = st.number_input('BMI', min_value=0.0, max_value=100.0, step=0.1)
    smoker = st.selectbox('Smoker', ['No', 'Yes'])
    expenses = st.number_input('Expenses', min_value=0.0, max_value=100000.0, step=0.01)
    region_northeast = st.checkbox('Region Northeast')
    region_northwest = st.checkbox('Region Northwest')
    region_southeast = st.checkbox('Region Southeast')
    region_southwest = st.checkbox('Region Southwest')
    bodymass_category_Normal = st.checkbox('Body Mass Category Normal')
    bodymass_category_Obesity = st.checkbox('Body Mass Category Obesity')
    bodymass_category_Overweight = st.checkbox('Body Mass Category Overweight')
    bodymass_category_Thin = st.checkbox('Body Mass Category Thin')
    button = st.button('Clust!')
    smoker_numeric = 1 if smoker == 'Yes' else 0
    user_data = pd.DataFrame({
        'age': [age],
        'bmi': [bmi],
        'smoker': [smoker_numeric],
        'expenses': [expenses],
        'region_northeast': [region_northeast],
        'region_northwest': [region_northwest],
        'region_southeast': [region_southeast],
        'region_southwest': [region_southwest],
        'bodymass_category_Normal': [bodymass_category_Normal],
        'bodymass_category_Obesity': [bodymass_category_Obesity],
        'bodymass_category_Overweight': [bodymass_category_Overweight],
        'bodymass_category_Thin': [bodymass_category_Thin]
    })
    if button:
        with open('kmeans.pkl', 'rb') as file:
            loaded_model = pickle.load(file)
        
        predicted = loaded_model.predict(user_data)
        st.write(predicted)

def klasifikasi(df):
    x_final = df.drop("sex", axis=1) 
    scaler = MinMaxScaler()
    x_final_norm = scaler.fit_transform(x_final)
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(x_final_norm)
    kmeans_clust = kmeans.predict(x_final_norm)
    combined_data_assoc = pd.concat([df.reset_index(drop=True), pd.DataFrame(kmeans_clust, columns=["kmeans_cluster"])], axis=1)

    # Visualisasi dengan histplot
    plt.figure(figsize=(10, 6))
    for cluster in sorted(combined_data_assoc["kmeans_cluster"].unique()):
        sns.histplot(combined_data_assoc[combined_data_assoc["kmeans_cluster"] == cluster]["age"], kde=True, label=f'Cluster {cluster}', alpha=0.5)
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    plt.title('Histogram of Age by K-Means Cluster')
    plt.legend()
    st.pyplot(plt)
    text = """
    Fungsi `klasifikasi` dan `clustering` ini bertujuan untuk melakukan clustering data menggunakan algoritma K-Means. Berikut penjelasan masing-masing bagian:

    - **klasifikasi(df)**:
    Fungsi ini melakukan klasifikasi data dengan algoritma K-Means. Pertama, fitur target 'sex' dihapus dari dataset untuk mempersiapkan data untuk proses clustering. Data kemudian dinormalisasi menggunakan Min-Max Scaler. Selanjutnya, model K-Means dengan 3 kluster diterapkan pada data yang sudah dinormalisasi. Hasil clustering kemudian disertakan ke dalam dataset dengan nama kolom 'kmeans_cluster'. Fungsi ini juga memvisualisasikan histogram usia (Age) berdasarkan kluster yang dihasilkan oleh K-Means.

    - **clustering(df)**:
    Fungsi ini memungkinkan pengguna untuk memasukkan data dan melakukan prediksi kluster dengan model K-Means yang telah dilatih sebelumnya. Pengguna diminta untuk memasukkan beberapa fitur seperti usia (Age), indeks massa tubuh (BMI), status merokok (Smoker), biaya (Expenses), serta opsi untuk memilih beberapa atribut lain seperti wilayah geografis (Region) dan kategori indeks massa tubuh (Body Mass Category). Setelah memasukkan data, pengguna dapat menekan tombol "Clust!" untuk melakukan prediksi kluster menggunakan model K-Means. Hasil prediksi kluster akan ditampilkan.

    """
    st.markdown(text)