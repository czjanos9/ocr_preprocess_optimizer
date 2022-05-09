Tesseract telepítési helye:
'C:\Program Files\Tesseract-OCR\tesseract.exe'
A forráskód elején lehet módosítani, ha szükséges.

Függöségek:
Tesseract letöltés: https://github.com/UB-Mannheim/tesseract/wiki

Python 3.7

OpenCV
pip install opencv-python

PyTesseract
pip install pytesseract

scikit learn
pip install -U scikit-learn

Parancssorból futtatható:
python ocr_optimizer_by_preprocess.py

MENÜ:
[0] Manuális vezérlés tesztelése

[1] Adatok generálása (paramétertér lefedése, sok időt vehet igénybe.)

[2] A paramétertérből TData osztály készítése és annak mentése fáljba (van egy előre elkészített)

[3] Az adatokat felhasználva és egy új képet megadva paraméterezést állít be és karakterfelismerést hajt végre.

[4] Program leállítása 