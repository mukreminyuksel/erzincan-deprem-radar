"""Streamlit Community Cloud giriş noktası (entry point).

Gerçek uygulama ``earthquake.py`` dosyasındadır. Streamlit Cloud bu repo için
ana dosya olarak ``deprem.py``'ye ayarlı olduğundan, bu ince katman (shim)
``earthquake.py``'yi her yeniden çalıştırmada (rerun) script olarak yürütür.

Not: ``import earthquake`` KULLANILMAZ — Python modülü bir kez import edip
cache'ler, oysa Streamlit her etkileşimde ana dosyanın baştan çalışmasını ister.
``runpy.run_path`` bunu garanti eder.
"""

import runpy

runpy.run_path("earthquake.py", run_name="__main__")
