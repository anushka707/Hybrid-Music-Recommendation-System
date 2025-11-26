from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from .config import ALS_MODEL_DIR, DATA_RAW

class ALSRecommender:
    def __init__(self):
        self.spark = (
            SparkSession.builder
            .appName("LastFM_ALS")
            .master("local[*]")
            .getOrCreate()
        )

    def load_data(self):
        df = self.spark.read.csv(
            f"{DATA_RAW}/user_artists.dat",
            sep="\t",
            header=True,
            inferSchema=True
        )
        return df.select("userID", "artistID", "weight")

    def train(self):
        df = self.load_data()

        train, test = df.randomSplit([0.8, 0.2], seed=42)

        als = ALS(
            maxIter=15,
            regParam=0.1,
            rank=20,
            userCol="userID",
            itemCol="artistID",
            ratingCol="weight",
            coldStartStrategy="drop",
            nonnegative=True
        )

        self.model = als.fit(train)

        evaluator = RegressionEvaluator(
            metricName="rmse",
            labelCol="weight",
            predictionCol="prediction"
        )
        rmse = evaluator.evaluate(self.model.transform(test))
        print(f"\n📉 RMSE on Test Set: {rmse:.4f}\n")

        (
            self.model.write()
            .overwrite()
            .save(ALS_MODEL_DIR)
        )

        return self.model

    def recommend_for_user(self, user_id, num_items=5):
        user_df = self.spark.createDataFrame([(user_id,)], ["userID"])
        return self.model.recommendForUserSubset(user_df, num_items)
