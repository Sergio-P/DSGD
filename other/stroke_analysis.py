import pandas as pd


data = pd.read_csv("data/stroke_data_2.csv")

# No stroke
# df = data[(data["WBC"] > 10000) & (data["Ht"] < 38)]
# df = data[(data["CREA"] > 1.2) & (data["GGT"] > 30)]

# Stroke
# df = data[(data["UA"] > 7) & (data["bmis"] < 18)]
df = data[(data["HD_cardiovascullar"] == 1)]

print("Total cases: %d of %d (%.1f%%)" % (len(df), len(data), 100. * len(df) / len(data)))
vc = df.cls.value_counts()
print("No stroke:\t%.2f%%" % (100.0 * vc[0] / len(df)))
print("Stroke:\t\t%.2f%%" % (100.0 * vc[1] / len(df)))

