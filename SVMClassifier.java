package com.company;
/*
 * Name: Angitha Mathew
 * sentiment analysis using classification and evaluating the prediction
 */

import java.util.Arrays;
import java.util.List;
import java.util.regex.Pattern;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.classification.SVMModel;
import org.apache.spark.mllib.classification.SVMWithSGD;
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.feature.HashingTF;
import org.apache.spark.mllib.feature.IDF;
import org.apache.spark.mllib.feature.IDFModel;
import org.apache.spark.mllib.linalg.Vector;
import scala.Tuple2;

/**
 * STEP 1: convert Split the lines into sentences and it's corresponding labels.
 * //STEP 2:Calculate term Frequency or any other appropriate vector like TF-IDF or word embeddings that can be passed to the SVM model
 * //STEP 3: Create the model and train it using 60% of the data
 * //STEP 4: Predit the labels for the remaining 40%
 * STEP 5: Print some of the predictions and evaluate the model using AUROC
 **/

public final class SVMClassifier { // using MapReduce (sort of)
    private static final Pattern SPACE = Pattern.compile(" ");

    public static void main(String[] args) throws Exception {
        System.setProperty("hadoop.home.dir", "C:/winutils");//so that we can avoid setting the environment variable HADOOP_HOME manually
        SparkConf sparkConf = new SparkConf().setAppName("WordCount")//sets the spark configuration with default values
                // expect for the ones manually set like AppName
                .setMaster("local[4]").set("spark.executor.memory", "1g");
        //PLEASE CHANGE THIS PATH BEFORE TRYING TO RUN!!
        String inputFileName = "D:\\MAI-SEMESTER-1\\LSDA\\imdb_labelled.txt";// reading the file from the path

        JavaSparkContext ctx = new JavaSparkContext(sparkConf);
        JavaRDD<String> linesRDD = ctx.textFile(inputFileName, 1);//input file is read and distributed as lines

        /*
        The lines are taken one at a time, split first on the basis of tab space to seperate the label and the sentence
        and then on the basis of white spaces to get list of strings.
        These list of strings is passed into HashingTF to get term frequencies
         */
        JavaRDD<Tuple2<Integer, Vector>> labelVectorRDD = linesRDD.map((String s) -> {
            String[] sentenceAndLabels = s.split("\t");//seperating labels
            Integer true_label = Integer.valueOf(sentenceAndLabels[1]);// storing value of labels
            List<String> wordsOfEachLine = Arrays.asList(sentenceAndLabels[0].split(" "));
            HashingTF hTFModel = new HashingTF();// creating a Term Frequency model since we can't pass sentences to the SVM Classifier.
            Vector termFrequency = hTFModel.transform(wordsOfEachLine);// term Frequency vector for each sentence
            return new Tuple2<Integer, Vector>(true_label, termFrequency);// the label and the vector is returned
        });


        //Input or parameter passed to SVM model is of type JavaRDD<LabeledPoint>, hence the tuple is used to create the same
        JavaRDD<LabeledPoint> LabeledPointRDD = labelVectorRDD.map(f -> new LabeledPoint(f._1(), f._2()));
        JavaRDD<LabeledPoint> trainingLP = LabeledPointRDD.sample(false, 0.6, 11L);//using 60% of the data for training
        trainingLP.cache();
        JavaRDD<LabeledPoint> testLP = LabeledPointRDD.subtract(trainingLP);// using the rest for testing
        // Run training algorithm to build the model.
        int numIterations = 500;
        SVMModel model = SVMWithSGD.train(trainingLP.rdd(), numIterations);// specifying the number of epochs to get better results.

        // Clear the default threshold.
        model.clearThreshold();

        // Using Model.predict to compute the raw scores which are then used to label the sentences
        JavaRDD<Tuple2<Object, Object>> rawScoreAndLabels_test = testLP.map(p ->
                new Tuple2<>(model.predict(p.features()), p.label()));

        int count = 25;// To print 25 samples of test data with it's raw score and label
        for (Tuple2<Object, Object> eachSen : rawScoreAndLabels_test.collect()) {
            System.out.println("Score for test data " + count + ": " + eachSen._1() + " and it's label: " + eachSen._2());
            count--;// Here the raw scores are printed and there is a threshold for classification into 0 or 1.
            if (count == 0)
                break;
        }
        /** Get evaluation metrics. BinaryClassificationMetrics has predefined methods to calculate  area under the precision-recall curve
         F-Measure by threshold, precision, recall etc including the Area under receiver operating characteristic (ROC) curve.
         **/

        BinaryClassificationMetrics metrics_test =
                new BinaryClassificationMetrics(JavaRDD.toRDD(rawScoreAndLabels_test));// we pass the predications and labels to the class

        double auROC_test = metrics_test.areaUnderROC();// the metric value is returned.
        //Printing the evaluation metric. Here the entirety of the test data is used for calculating AuRoC
        System.out.println("Area under ROC = " + auROC_test);

        ctx.stop();
        ctx.close();
    }
}

