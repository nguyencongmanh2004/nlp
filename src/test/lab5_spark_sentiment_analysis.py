from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression, NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, HashingTF, IDF, Word2Vec

# scikit-learn
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import numpy as np

def main():
    spark = SparkSession.builder.appName("Sentiment Analysis").getOrCreate()
    data_path = "src/data/sentiments.csv"
    df = spark.read.csv(data_path, header=True, inferSchema=True)

    # Chuẩn hóa label (-1,1) -> (0,1)
    df = df.withColumn("label", ((col("sentiment").cast("integer") + 1) / 2).cast("integer"))
    df = df.dropna(subset=["sentiment"])

    # --- Feature preprocessing ---
    processor = createPreprocessor(df)
    df = processor.transform(df)

    # --- Train/test split ---
    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

    # --- Naive Bayes ---
    naive = NaiveBayes(featuresCol="parse_vector", labelCol="label")
    nb_model = naive.fit(train_df)
    nb_pred = nb_model.transform(test_df)
    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
    acc_nb = evaluator.evaluate(nb_pred)
    print(f"Naive Bayes Accuracy: {acc_nb:.4f}")

    # --- Logistic Regression ---
    lr_dense = LogisticRegression(featuresCol="dense_vector", labelCol="label", maxIter=50, regParam=0.01)
    lr_sparse = LogisticRegression(featuresCol="parse_vector", labelCol="label", maxIter=50, regParam=0.01)
    lr_dense_model = lr_dense.fit(train_df)
    lr_sparse_model = lr_sparse.fit(train_df)

    lr_dense_pred = lr_dense_model.transform(test_df)
    lr_sparse_pred = lr_sparse_model.transform(test_df)
    acc_lr_dense = evaluator.evaluate(lr_dense_pred)
    acc_lr_sparse = evaluator.evaluate(lr_sparse_pred)
    print(f"Logistic Regression (dense): {acc_lr_dense:.4f}")
    print(f"Logistic Regression (sparse): {acc_lr_sparse:.4f}")

    # --- MLP scikit-learn ---
    run_mlp_sklearn(df, features_col="dense_vector")

    spark.stop()


def createPreprocessor(train_df: DataFrame):
    tokenizer = RegexTokenizer(inputCol="text", outputCol="tokens", pattern=r'[! ,\-:;?)("\'\n]+', toLowercase=True)
    remover = StopWordsRemover(inputCol="tokens", outputCol="filtered_tokens")
    w2v = Word2Vec(vectorSize=512, minCount=3, windowSize=3, inputCol="filtered_tokens", outputCol="dense_vector")
    tf = HashingTF(inputCol="filtered_tokens", outputCol="raw_features", numFeatures=5000)
    idf = IDF(inputCol="raw_features", outputCol="parse_vector")

    pipeline = Pipeline(stages=[tokenizer, remover, w2v, tf, idf])
    return pipeline.fit(train_df)


def run_mlp_sklearn(df, features_col="dense_vector"):
    # Chuyển sang pandas
    data = df.select(features_col, "label").toPandas()
    X = np.array(data[features_col].tolist())
    y = np.array(data["label"].tolist())

    # Chia train/test
    n = X.shape[0]
    idx = np.random.permutation(n)
    train_idx = idx[:int(0.8*n)]
    test_idx  = idx[int(0.8*n):]

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test   = X[test_idx], y[test_idx]

    # --- MLP 2 hidden layers 32 32 ---
    clf = MLPClassifier(hidden_layer_sizes=(32,32), activation='relu', solver='adam', max_iter=300)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"MLP (scikit-learn) Accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()
