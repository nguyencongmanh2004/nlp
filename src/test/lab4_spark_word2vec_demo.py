import re
from pyspark.sql import SparkSession
from pyspark.ml.feature import Word2Vec
from pyspark.sql.functions import col, lower, regexp_replace, split

def main():
    spark = SparkSession.builder \
        .appName("Word2Vec_Spark_Demo") \
        .master("local[*]") \
        .getOrCreate()

    print("Spark session started.")

    # Đọc dữ liệu JSON
    data_path = "data/c4-train.00000-of-01024-30K.json"
    print(f"Loading data from: {data_path}")

    df = spark.read.json(data_path)
    print(f" Loaded {df.count()} rows.")

    # Giữ lại cột "text"
    df = df.select("text").na.drop()

    # tiền xử lý dữ liệu
    # - Chuyển về chữ thường
    # - Loại bỏ ký tự đặc biệt và dấu câu
    # - Tách thành danh sách các từ (array<string>)
    df_clean = df.withColumn("text", lower(col("text")))
    df_clean = df_clean.withColumn("text", regexp_replace(col("text"), r"[^a-z\s]", " "))
    df_clean = df_clean.withColumn("words", split(col("text"), r"\s+"))

    print("Text preprocessing completed.")
    df_clean.show(3, truncate=100)

    # Cấu hình và huấn luyện Word2Vec
    print("Training Word2Vec model...")
    word2vec = Word2Vec(
        vectorSize=100,    # Kích thước embedding
        minCount=2,        # Bỏ từ xuất hiện ít hơn 2 lần
        inputCol="words",
        outputCol="result"
    )

    model = word2vec.fit(df_clean)
    print("model training completed.")

    # Dùng thử model
    test_word = "computer"
    try:
        synonyms = model.findSynonyms(test_word, 5)
        print(f"\Top 5 words similar to '{test_word}':")
        synonyms.show(truncate=False)
    except Exception as e:
        print(f"Cannot find synonyms for '{test_word}': {e}")

    # Dừng Spark
    spark.stop()
    print("Spark session stopped.")

if __name__ == "__main__":
    main()
