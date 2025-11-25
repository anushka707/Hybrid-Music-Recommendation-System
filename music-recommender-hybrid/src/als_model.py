import os
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from .config import DATA_RAW, ALS_MODEL_DIR


class ALSRecommender:
    def __init__(self, rank=20, maxIter=15, regParam=0.1):
        self.rank = rank
        self.maxIter = maxIter
        self.regParam = regParam
        self.spark = (
            SparkSession.builder
            .appName("MusicRecommenderALS")
            .config("spark.driver.memory", "8g")
            .config("spark.executor.memory", "8g")
            .getOrCreate()
        )
        self.model = None

    def load_interaction_data(self):
        """
        user_artists.dat format:
        userID   artistID   weight (listen_count)
        """
        df = self.spark.read.csv(
            os.path.join(DATA_RAW, "user_artists.dat"),
            sep="\t",
            header=True,
            inferSchema=True
        )

        df = df.withColumnRenamed("weight", "listen_count")

        return df

    def train(self):
        df = self.load_interaction_data()

        train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)

        als = ALS(
            userCol="userID",
            itemCol="artistID",
            ratingCol="listen_count",
            maxIter=self.maxIter,
            regParam=self.regParam,
            rank=self.rank,
            coldStartStrategy="drop",
            nonnegative=True
        )

        self.model = als.fit(train_data)

        # Evaluate
        evaluator = RegressionEvaluator(
            metricName="rmse", 
            labelCol="listen_count", 
            predictionCol="prediction"
        )

        predictions = self.model.transform(test_data)
        rmse = evaluator.evaluate(predictions)

        print(f"\n📉 RMSE on Test Set: {rmse:.4f}\n")

        # Save model
        self.model.save(ALS_MODEL_DIR)

    def load_model(self):
        from pyspark.ml.recommendation import ALSModel
        self.model = ALSModel.load(ALS_MODEL_DIR)

    def recommend_for_user(self, user_id, k=10):
        """
        Return recommended artist IDs for a user.
        """
        if self.model is None:
            self.load_model()

        user_df = self.spark.createDataFrame([(user_id,)], ["userID"])
        recs = self.model.recommendForUserSubset(user_df, k)
        return recs

    def recommend_for_all_users(self, k=10):
        if self.model is None:
            self.load_model()
        return self.model.recommendForAllUsers(k)
