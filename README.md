# oral-diseases

Data conversion
Project Skilvul Challenge Cycle 2

Domain ID dapat di https://learn.microsoft.com/en-us/azure/ai-services/custom-vision-service/select-domain

## How to Run

1. Download file autentikasi kaggle
    - Buka kaggle.com
    - Klik gambar profil
    - Pilih settings
    - Scroll kebawah sampai ketemu section API
    - Klik "Create New Token"
    - Simpan file kaggle.json di folder ~/.kaggle/
      - contoh nama user "ucok":
        - /home/ucok/.kaggle/kaggle.json (linux)
        - C:\Users\ucok\\.kaggle\kaggle.json (windows)
2. Ganti nama `.env.example` menjadi `.env`
3. Isi file `.env` dengan key berikut. Valuenya lihat di settings project di customvision.ai
    ```dotenv
    VISION_TRAINING_KEY=<YOUR_TRAINING_KEY>
    VISION_TRAINING_ENDPOINT=<YOUR_TRAINING_ENDPOINT>
    DOMAIN_ID=<SESUAIKAN_DENGAN_NAMA_DOMAIN_DI_ATAS>
    PROJECT_ID=<ID_PROJECT>
    ```
4. Instal requirements (disarankan menggunakan [virtualenv](https://python.land/virtual-environments/virtualenv))
    ```bash
    pip install -r requirements.txt
    ```
5. Jalankan main.py
