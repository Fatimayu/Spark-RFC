import org.apache.spark.ml.regression.LinearRegression;
import org.apache.spark.ml.regression.LinearRegressionModel;
import org.apache.spark.ml.regression.LinearRegressionTrainingSummary;
import org.apache.spark.ml.regression.*;
import org.apache.spark.ml.linalg.Vectors;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.*;
import org.apache.spark.storage.*;
import org.apache.spark.sql.functions.*;
import org.apache.spark.sql.types.*;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.RandomForestClassificationModel;
import org.apache.spark.ml.classification.RandomForestClassifier;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.*;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.mllib.evaluation.MulticlassMetrics;

import org.apache.spark.ml.*;

public class MakePrediction{
    
    public static void main(String[] args){
        
        SparkSession spark = SparkSession
  .builder()
  .appName("Java Spark SQL basic example")
  .config("spark.master", "local")
  .getOrCreate();
/*// Load training data.
Dataset<Row> ds = spark.read().format("csv").option("header","true").option("delimiter", ";").load("TrainingDataset.csv");
ds =   ds.withColumn("fixed acidity", ds.col("fixed acidity").cast(DataTypes.DoubleType));
ds =   ds.withColumn("volatile acidity", ds.col("volatile acidity").cast(DataTypes.DoubleType));
ds =   ds.withColumn("citric acid", ds.col("citric acid").cast(DataTypes.DoubleType));
ds =   ds.withColumn("residual sugar", ds.col("residual sugar").cast(DataTypes.DoubleType));
ds =   ds.withColumn("chlorides", ds.col("chlorides").cast(DataTypes.DoubleType));
ds =   ds.withColumn("free sulfur dioxide", ds.col("free sulfur dioxide").cast(DataTypes.DoubleType));
ds =   ds.withColumn("total sulfur dioxide", ds.col("total sulfur dioxide").cast(DataTypes.DoubleType));
ds =   ds.withColumn("density", ds.col("density").cast(DataTypes.DoubleType));  
ds =   ds.withColumn("pH", ds.col("pH").cast(DataTypes.DoubleType));
ds =   ds.withColumn("sulphates", ds.col("sulphates").cast(DataTypes.DoubleType));
ds =   ds.withColumn("alcohol", ds.col("alcohol").cast(DataTypes.DoubleType));
ds =   ds.withColumn("quality", ds.col("quality").cast(DataTypes.IntegerType));

        
          VectorAssembler assembler = new VectorAssembler()
    .setInputCols(new String[]{"volatile acidity","chlorides","total sulfur dioxide","density","pH","sulphates","alcohol"})
    .setOutputCol("features");
        
           ds.printSchema();
        System.out.println("------------------Printed Schema-------------------------");
      
  
Dataset<Row> trainingData = assembler.transform(ds);

        
        StringIndexerModel labelIndexer = new StringIndexer()
  .setInputCol("quality")
  .setOutputCol("indexedLabel")
  .fit(trainingData);
// Automatically identify categorical features, and index them.
// Set maxCategories so features with > 4 distinct values are treated as continuous.
VectorIndexerModel featureIndexer = new VectorIndexer()
  .setInputCol("features")
  .setOutputCol("indexedFeatures")
  .setMaxCategories(50)
  .fit(trainingData);
        




// Train a RandomForest model.
RandomForestClassifier rf = new RandomForestClassifier()
  .setLabelCol("indexedLabel")
  .setFeaturesCol("indexedFeatures");

// Convert indexed labels back to original labels.
IndexToString labelConverter = new IndexToString()
  .setInputCol("prediction")
  .setOutputCol("predictedLabel")
  .setLabels(labelIndexer.labelsArray()[0]);

// Chain indexers and forest in a Pipeline
Pipeline pipeline = new Pipeline()
  .setStages(new PipelineStage[] {labelIndexer, featureIndexer, rf, labelConverter});

// Train model. This also runs the indexers.
PipelineModel model = pipeline.fit(trainingData);
        try{
model.save( "rfc_model.model");
        }catch(Exception e){}
        
        */
        
        
        
PipelineModel model = PipelineModel.load("rfc_model.model");
        
 Dataset<Row> validation = spark.read().format("csv").option("header","true").option("delimiter", ";").load("ValidationDataset.csv");
validation =   validation.withColumn("fixed acidity", validation.col("fixed acidity").cast(DataTypes.DoubleType));
validation =   validation.withColumn("volatile acidity", validation.col("volatile acidity").cast(DataTypes.DoubleType));
validation =   validation.withColumn("citric acid", validation.col("citric acid").cast(DataTypes.DoubleType));
validation =   validation.withColumn("residual sugar", validation.col("residual sugar").cast(DataTypes.DoubleType));
validation =   validation.withColumn("chlorides", validation.col("chlorides").cast(DataTypes.DoubleType));
validation =   validation.withColumn("free sulfur dioxide", validation.col("free sulfur dioxide").cast(DataTypes.DoubleType));
validation =   validation.withColumn("total sulfur dioxide", validation.col("total sulfur dioxide").cast(DataTypes.DoubleType));
validation =   validation.withColumn("density", validation.col("density").cast(DataTypes.DoubleType));  
validation =   validation.withColumn("pH", validation.col("pH").cast(DataTypes.DoubleType));
validation =   validation.withColumn("sulphates", validation.col("sulphates").cast(DataTypes.DoubleType));
validation =   validation.withColumn("alcohol", validation.col("alcohol").cast(DataTypes.DoubleType));
validation =   validation.withColumn("quality", validation.col("quality").cast(DataTypes.IntegerType));       
        
        
       VectorAssembler assembler2 = new VectorAssembler()
    .setInputCols(new String[]{"volatile acidity","chlorides","total sulfur dioxide","density","pH","sulphates","alcohol"})
    .setOutputCol("features");
        
          validation.printSchema();
        Dataset<Row> testData = assembler2.transform(validation);
    
  

// Make predictions.
Dataset<Row> predictions = model.transform(testData);

// Select example rows to display.
predictions.select("predictedLabel", "quality", "features").show(50);

// Select (prediction, true label) and compute test error
MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
  .setLabelCol("quality")
  .setPredictionCol("predictedLabel")
  .setMetricName("accuracy");
predictions =   predictions.withColumn("predictedLabel", predictions.col("predictedLabel").cast(DataTypes.DoubleType));

MulticlassMetrics metrics = evaluator.getMetrics(predictions);
        double metricLabel = evaluator.getMetricLabel();
        double[] labels = metrics.labels();
double accuracy = metrics.accuracy();
double f1Sum = 0;
System.out.println("\n\n\n\n\n\n--------------Training metrics------------------");
System.out.println("Accuracy = " + accuracy);
System.out.println("F1 scores:");
        for(double d: labels){
            
            System.out.println("Quality - " + d + "\t F1 score - " + metrics.fMeasure(d));
            f1Sum+=metrics.fMeasure(d);
        }
System.out.println("Average F1 score - " + f1Sum/labels.length);  
System.out.println("---------------------------------------------------------\n\n\n\n\n\n");
RandomForestClassificationModel rfModel = (RandomForestClassificationModel)(model.stages()[2]);
System.out.println("Learned classification forest model:\n" + rfModel.toDebugString());
        
        
        
/*
LinearRegression lr = new LinearRegression()
.setLabelCol("quality")
  .setMaxIter(1000000)
  .setRegParam(0.1)
  .setElasticNetParam(0.1);

// Fit the model.
LinearRegressionModel lrModel = lr.fit(training);

System.out.println(lrModel.transform(test).select("features", "quality", "prediction").getClass());
        System.out.println("Object Type");
lrModel.transform(test).select("features", "quality", "prediction").show();
        
        
        
// Print the coefficients and intercept for linear regression.
System.out.println("Coefficients: "
  + lrModel.coefficients() + " Intercept: " + lrModel.intercept());

// Summarize the model over the training set and print out some metrics.
LinearRegressionTrainingSummary trainingSummary = lrModel.summary();
System.out.println("numIterations: " + trainingSummary.totalIterations());
System.out.println("objectiveHistory: " + Vectors.dense(trainingSummary.objectiveHistory()));
trainingSummary.residuals().show();
System.out.println("RMSE: " + trainingSummary.rootMeanSquaredError());
System.out.println("r2: " + trainingSummary.r2());
        
//test run summary
LinearRegressionSummary summary = lrModel.evaluate(test);
System.out.println("\n\n\n\n\nTesting Data R2: " + summary.r2());
System.out.println("\n\n\n\n\nTesting Data RMSE: " + summary.rootMeanSquaredError());

        
        

}
}
**/
    }}