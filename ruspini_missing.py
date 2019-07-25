#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#impyute digunakan untuk memmasukkan nilai yg hilang
import impyute as impy

data_ruspini = pd.read_csv("data_ruspini_missing.csv")
data_ruspini = data_ruspini.replace("?", np.nan)
#data_ruspini

data_ruspini_array = np.array(data_ruspini, dtype=float)
data_baru = impy.mean(data_ruspini_array)
#data_baru

data_frame_ruspini_missing = pd.DataFrame({
    'x': data_ruspini_array[:, 0],
    'y': data_ruspini_array[:, 1],
    'label': data_ruspini_array[:, 2],
})

data_frame_ruspini_baru = pd.DataFrame({
    'x': data_baru[:, 0],
    'y': data_baru[:, 1],
    'label': data_baru[:, 2],
})

print(data_frame_ruspini_baru)
print(data_frame_ruspini_missing)

#visualisasi
plt.figure('Ruspini Missing')
plt.scatter(data_frame_ruspini_missing['x'].values,
            data_frame_ruspini_missing['y'].values,
            color='b')
plt.xlabel('X')
plt.ylabel('Y')

plt.figure('Ruspini Baru')
plt.scatter(data_frame_ruspini_baru['x'].values,
            data_frame_ruspini_baru['y'].values,
            color='blue')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

