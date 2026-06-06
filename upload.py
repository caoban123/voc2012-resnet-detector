import os
import kagglehub

os.environ["KAGGLE_API_TOKEN"] = "KGAT_3ceeddf86bf1081fd1d99387d134819f"
os.environ["KAGGLEHUB_CACHE"] = r"D:\football\data"

path = kagglehub.dataset_download("banuprasadb/pascal-voc-2012")

print("\nTải thành công! Bộ dữ liệu đã nằm tại:")
print(path)