# Ajan 3 — Vektörleştirme Uzmanı
**Rol:** Python döngülerini NumPy vektör operasyonlarına dönüştürür.
**Uzmanlık:** NumPy, pandas vectorization, haversine formülü, sliding-window algoritmaları.

## Uygulanan Optimizasyonlar (v1.13)

### Haversine Vectorization (28× hızlanma)
```python
# ÖNCE (yavaş):
df["uzaklik_km"] = df.apply(lambda row: haversine(lat, lon, row.lat, row.lon), axis=1)

# SONRA (hızlı):
lat_rad = math.radians(lat)
lon_rad_val = math.radians(lon)
df_lat_rad = np.radians(df["lat"].to_numpy())
df_lon_rad = np.radians(df["lon"].to_numpy())
dlat = df_lat_rad - lat_rad
dlon = df_lon_rad - lon_rad_val
a_hav = np.sin(dlat/2)**2 + math.cos(lat_rad)*np.cos(df_lat_rad)*np.sin(dlon/2)**2
df["uzaklik_km"] = 6371.0 * 2.0 * np.arcsin(np.sqrt(a_hav))
```

### Sliding-Window Dedup (O(n²) → O(n×k))
- Zaman dizisi numpy array'e çekilir
- Binary search ile pencere bulunur
- k = pencere boyutu (sabit), n = toplam kayıt sayısı

## Referans Dosya
- `earthquake_core.py` → `nearest_fault_vertex_distance()` — bbox pre-filter eklendi
