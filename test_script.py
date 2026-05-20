import os
print("Current working directory:", os.getcwd())
print("Files in data directory:", os.listdir("data/"))
print("Does heart_disease.csv exist?", os.path.exists("data/heart_disease.csv"))