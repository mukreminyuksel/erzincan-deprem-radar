"""turkey_faults.geojson'daki bozuk karakteri temizler.

MTA verisinin dönüştürülmesinde Ğ ve Ş harflerinden sonra fazladan bir '\\x9e'
kontrol karakteri kalmış (Windows-1254 kalıntısı): "DOĞ\\x9eU ANADOLU FAY ZONU",
"AKŞ\\x9eEHİR". Haritadaki fay açıklamalarında bozuk görünüyor ve aynı fay zonu
iki farklı adla sayılıyordu. Yalnız metin alanları değişir; koordinatlara dokunulmaz.

Çalıştırma: python tools/fix_fault_names.py
"""
import json
from pathlib import Path

PATH = Path(__file__).resolve().parent.parent / "turkey_faults.geojson"


def main():
    gj = json.loads(PATH.read_text(encoding="utf-8"))
    fixed = 0
    for feat in gj["features"]:
        props = feat["properties"]
        for key, val in props.items():
            if isinstance(val, str) and "\x9e" in val:
                props[key] = val.replace("\x9e", "")
                fixed += 1
    PATH.write_text(json.dumps(gj, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"{fixed} metin alanı düzeltildi")


if __name__ == "__main__":
    main()
