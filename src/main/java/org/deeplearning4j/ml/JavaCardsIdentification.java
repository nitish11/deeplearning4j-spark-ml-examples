package org.deeplearning4j.ml;

import org.apache.spark.SparkConf;
import org.apache.spark.SparkContext;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.feature.StandardScaler;
import org.apache.spark.sql.DataFrame;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SQLContext;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.conf.layers.setup.ConvolutionLayerSetup;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.spark.ml.classification.NeuralNetworkClassification;
import org.deeplearning4j.spark.sql.sources.mnist.DefaultSource;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.util.FeatureUtil;

import java.util.HashMap;
import java.util.Map;

/**
 * Program to detect cards from the frames obtained from a video
 * Modified By : Nitish Bhardwaj
 * Last Modified : 22nd September
 * Run with
 * <pre>
 * bin/run-example ml.JavaCardsIdentification
 * </pre>
 */
public class JavaCardsIdentification {

    final static int outputNum = 10;

    public static void main(String[] args) {
        SparkConf conf = new SparkConf().setMaster("local[*]")
                .setAppName("Cards Identification Pipeline (Java)");
        SparkContext jsc = new SparkContext(conf);
        SQLContext jsql = new SQLContext(jsc);

        String imagesPath =  "file://" + System.getProperty("user.home") + "/MNIST/images-idx1-ubyte";
        String labelsPath =  "file://" + System.getProperty("user.home") + "/MNIST/labels-idx1-ubyte";
        Map<String, String> params = new HashMap<>();
        params.put("imagesPath", imagesPath);
        params.put("labelsPath", labelsPath);
        params.put("recordsPerPartition", "400");
        params.put("maxRecords", "2000");

        DataFrame data = jsql.read().format(DefaultSource.class.getName())
                .options(params).load();

        System.out.println("\nLoaded Card Images dataframe:");
        data.show(100);

        DataFrame trainingData = data.sample(false, 0.8, 123);
        DataFrame testData = data.except(trainingData);

        StandardScaler scaler = new StandardScaler()
                .setInputCol("features")
                .setOutputCol("scaledFeatures");
        NeuralNetworkClassification classification = new NeuralNetworkClassification()
                .setFeaturesCol("scaledFeatures")
                .setEpochs(2)
                .setConf(getConfiguration());
        Pipeline pipeline = new Pipeline().setStages(new PipelineStage[]{
                scaler, classification});

        System.out.println("\nTraining...");
        PipelineModel model = pipeline.fit(trainingData);

        System.out.println("\nTesting...");
        DataFrame predictions = model.transform(testData);
        predictions.cache();

        System.out.println("\nTest Results:");
        predictions.show(100);

        Evaluation eval = new Evaluation(outputNum);
        Row[] rows = predictions.select("label","prediction").collect();
        for(int i = 0; i < rows.length; i++) {
            INDArray label = FeatureUtil.toOutcomeVector((int) rows[i].getDouble(0), outputNum);
            INDArray prediction = FeatureUtil.toOutcomeVector((int) rows[i].getDouble(1), outputNum);
            eval.eval(label, prediction);
        }

        System.out.println(eval.stats());
    }

    public static MultiLayerConfiguration getConfiguration() {

        final int numRows = 128;
        final int numColumns = 128;
        int nChannels = 1;
        int outputNum = 5;
        int batchSize = 10;
        int iterations = 5;
        int seed = 123;

        MultiLayerConfiguration.Builder builder = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .batchSize(batchSize)
                .iterations(iterations)
                .constrainGradientToUnitNorm(true)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .list(9)
                .layer(0, new ConvolutionLayer.Builder(128, 128)
                        .nIn(nChannels)
                        .nOut(8)
                        .weightInit(WeightInit.XAVIER)
                        .build())
                .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[]{2, 2})
                        .build())
                .layer(2, new ConvolutionLayer.Builder(10, 10)
                        .nIn(nChannels)
                        .nOut(6)
                        .weightInit(WeightInit.XAVIER)
                        .build())                        
                .layer(3, new ConvolutionLayer.Builder(10, 10)
                        .nIn(nChannels)
                        .nOut(6)
                        .weightInit(WeightInit.XAVIER)
                        .build())          
                .layer(4, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[]{2, 2})
                        .build())
                .layer(5, new ConvolutionLayer.Builder(10, 10)
                        .nIn(nChannels)
                        .nOut(6)
                        .weightInit(WeightInit.XAVIER)
                        .build())                        
                .layer(6, new ConvolutionLayer.Builder(10, 10)
                        .nIn(nChannels)
                        .nOut(6)
                        .weightInit(WeightInit.XAVIER)
                        .build())                        
                .layer(7, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[]{2, 2})
                        .build())
                        
                .layer(8, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nIn(150)
                        .nOut(outputNum)
                        .weightInit(WeightInit.XAVIER)
                        .activation("softmax")
                        .build())
                .backprop(true).pretrain(false);
        new ConvolutionLayerSetup(builder, numRows, numColumns, nChannels);

        MultiLayerConfiguration conf = builder.build();
        return conf;
    }
}
