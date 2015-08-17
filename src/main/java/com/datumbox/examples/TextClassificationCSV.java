/**
 * Copyright (C) 2013-2015 Vasilis Vryniotis <bbriniotis@datumbox.com>
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.datumbox.examples;

import com.datumbox.applications.nlp.TextClassifier;
import com.datumbox.common.dataobjects.Record;
import com.datumbox.common.persistentstorage.ConfigurationFactory;
import com.datumbox.common.persistentstorage.interfaces.DatabaseConfiguration;
import com.datumbox.common.utilities.PHPfunctions;
import com.datumbox.common.utilities.RandomGenerator;
import com.datumbox.framework.machinelearning.classification.MultinomialNaiveBayes;
import com.datumbox.framework.machinelearning.common.bases.mlmodels.BaseMLmodel;
import com.datumbox.framework.machinelearning.featureselection.categorical.ChisquareSelect;
import com.datumbox.framework.utilities.text.extractors.NgramsExtractor;
import java.net.URI;
import java.net.URISyntaxException;
import java.util.HashMap;
import java.util.Map;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import com.csvreader.CsvReader;
import com.csvreader.CsvWriter;
import au.com.bytecode.opencsv.CSVReader;
import au.com.bytecode.opencsv.CSVWriter;
import java.util.Arrays;
import java.util.List;
import java.io.FileWriter;

/**
 * Text Classification example.
 * 
 * @author Vasilis Vryniotis <bbriniotis@datumbox.com>
 */
public class TextClassificationCSV {
    
    /**
     * Example of how to use the TextClassifier class.
     * 
     * @param args the command line arguments
     * @throws java.net.URISyntaxException
     */
    public static void main(String[] args) throws URISyntaxException {        
        /**
         * There are two configuration files in the resources folder:
         * 
         * - datumbox.config.properties: It contains the configuration for the storage engines (required)
         * - logback.xml: It contains the configuration file for the logger (optional)
         */
        
        //Initialization
        //--------------
        RandomGenerator.setGlobalSeed(42L); //optionally set a specific seed for all Random objects
        DatabaseConfiguration dbConf = ConfigurationFactory.INMEMORY.getConfiguration(); //in-memory maps
        //DatabaseConfiguration dbConf = ConfigurationFactory.MAPDB.getConfiguration(); //mapdb maps
        
        
        
        //Reading Data
        //------------
        Map<Object, URI> dataset = new HashMap<>(); //The examples of each category are stored on the same file, one example per row.
        dataset.put("Computers & Electronics", TextClassificationCSV.class.getClassLoader().getResource("datasets/category-analysis/tech").toURI());
        dataset.put("Sports", TextClassificationCSV.class.getClassLoader().getResource("datasets/category-analysis/sports").toURI());
        dataset.put("Arts & Entertainment", TextClassificationCSV.class.getClassLoader().getResource("datasets/category-analysis/artEnter").toURI());
        dataset.put("Autos & Vehicles", TextClassificationCSV.class.getClassLoader().getResource("datasets/category-analysis/autoVehicle").toURI());     
        dataset.put("Beauty & Personal Care", TextClassificationCSV.class.getClassLoader().getResource("datasets/category-analysis/beauty").toURI());
        dataset.put("Business & Industry", TextClassificationCSV.class.getClassLoader().getResource("datasets/category-analysis/business").toURI());
        dataset.put("Books & Literature", TextClassificationCSV.class.getClassLoader().getResource("datasets/category-analysis/literate").toURI());
        dataset.put("Education", TextClassificationCSV.class.getClassLoader().getResource("datasets/category-analysis/education").toURI()); 

        dataset.put("Finance", TextClassificationCSV.class.getClassLoader().getResource("datasets/category-analysis/finance").toURI());     
        dataset.put("Food & Drink", TextClassificationCSV.class.getClassLoader().getResource("datasets/category-analysis/food").toURI());
        dataset.put("Games", TextClassificationCSV.class.getClassLoader().getResource("datasets/category-analysis/games").toURI());
        dataset.put("Health", TextClassificationCSV.class.getClassLoader().getResource("datasets/category-analysis/health").toURI());
        dataset.put("Home & Garden", TextClassificationCSV.class.getClassLoader().getResource("datasets/category-analysis/home").toURI());  
        dataset.put("Internet & Telecom", TextClassificationCSV.class.getClassLoader().getResource("datasets/category-analysis/internet").toURI());
        dataset.put("Law & Government", TextClassificationCSV.class.getClassLoader().getResource("datasets/category-analysis/law").toURI()); 
//++++++++++++++++++++

        dataset.put("Lifestyles", TextClassificationCSV.class.getClassLoader().getResource("datasets/category-analysis/lifestyle").toURI());     
        dataset.put("News", TextClassificationCSV.class.getClassLoader().getResource("datasets/category-analysis/news").toURI());
        dataset.put("Online Communities", TextClassificationCSV.class.getClassLoader().getResource("datasets/category-analysis/community").toURI());
        dataset.put("Pets & Animals", TextClassificationCSV.class.getClassLoader().getResource("datasets/category-analysis/pets").toURI());
        dataset.put("Real Estate", TextClassificationCSV.class.getClassLoader().getResource("datasets/category-analysis/realestate").toURI());
        dataset.put("Recreation", TextClassificationCSV.class.getClassLoader().getResource("datasets/category-analysis/recreation").toURI());  
        dataset.put("Reference & Language", TextClassificationCSV.class.getClassLoader().getResource("datasets/category-analysis/refNlang").toURI());
        dataset.put("Science", TextClassificationCSV.class.getClassLoader().getResource("datasets/category-analysis/science").toURI()); 
        dataset.put("Shopping", TextClassificationCSV.class.getClassLoader().getResource("datasets/category-analysis/shopping").toURI());     
        dataset.put("Travel", TextClassificationCSV.class.getClassLoader().getResource("datasets/category-analysis/travel").toURI());
        dataset.put("World Localities", TextClassificationCSV.class.getClassLoader().getResource("datasets/category-analysis/worldLocal").toURI());
        
        //Setup Training Parameters
        //-------------------------
        TextClassifier.TrainingParameters trainingParameters = new TextClassifier.TrainingParameters();
        
        //Classifier configuration
        trainingParameters.setMLmodelClass(MultinomialNaiveBayes.class);
        trainingParameters.setMLmodelTrainingParameters(new MultinomialNaiveBayes.TrainingParameters());
        
        //Set data transfomation configuration
        trainingParameters.setDataTransformerClass(null);
        trainingParameters.setDataTransformerTrainingParameters(null);
        
        //Set feature selection configuration
        trainingParameters.setFeatureSelectionClass(ChisquareSelect.class);
        trainingParameters.setFeatureSelectionTrainingParameters(new ChisquareSelect.TrainingParameters());
        
        //Set text extraction configuration
        trainingParameters.setTextExtractorClass(NgramsExtractor.class);
        trainingParameters.setTextExtractorParameters(new NgramsExtractor.Parameters());
        
        
        
        //Fit the classifier
        //------------------
        TextClassifier classifier = new TextClassifier("SentimentAnalysis", dbConf);
        classifier.fit(dataset, trainingParameters);
        
        
        
        //Use the classifier
        //------------------
        
        //Get validation metrics on the training set
        BaseMLmodel.ValidationMetrics vm = classifier.validate(dataset);
        classifier.setValidationMetrics(vm); //store them in the model for future reference
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//Read from CSV and classify:


	

	int count = 0;
	String inPath = "/home/milan/BitBucket/datumbox-framework-examples/src/main/resources/datasets/category-analysis/csvs/domain_categories_Active2.csv";
	String outPath = "/home/milan/BitBucket/datumbox-framework-examples/src/main/resources/datasets/category-analysis/csvs/all_categ.csv"; 

	try{
	CSVReader reader = new CSVReader(new FileReader(inPath), ',', '"', 1);
	CSVWriter writer = new CSVWriter(new FileWriter(outPath));
	String firstrow[] =  new String[3]; 
	firstrow[0] ="Domain";firstrow[1] ="Title";firstrow[2] ="Category";	
	writer.writeNext(firstrow);
		//CSVWriter writer = new CSVWriter(new FileWriter(outPath, true));
		
	      List<String[]> allRows = reader.readAll();
	       
	      //Read CSV line by line and use the string array as you want
	     for(String[] row : allRows){
		String sentence = row[1];
		Record r = classifier.predict(sentence);
		row[2] = ""+r.getYPredicted();
		writer.writeNext(row);
	     }
	}
	 catch (FileNotFoundException e) {
		e.printStackTrace();
	}

	 catch (IOException e) {
		e.printStackTrace();
	}

			

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++        
        //Classify a single sentence
        String sentence = "Latest Football News Transfer Rumours Goal.com";
        Record r = classifier.predict(sentence);
        
        System.out.println("Classifing sentence: \""+sentence+"\"");
        System.out.println("Predicted class: "+r.getYPredicted());
        System.out.println("Probability: "+r.getYPredictedProbabilities().get(r.getYPredicted()));
        
        System.out.println("Classifier Statistics: "+PHPfunctions.var_export(vm));
        
        
        sentence = "CarDekho - Cars in India, New Cars Prices 2015, Buy and Sell Used Cars";
        r = classifier.predict(sentence);
        
        System.out.println("Classifing sentence: \""+sentence+"\"");
        System.out.println("Predicted class: "+r.getYPredicted());
/*************************************************************************/
        sentence = "National Portal of India";
        r = classifier.predict(sentence);
        
        System.out.println("Classifing sentence: \""+sentence+"\"");
        System.out.println("Predicted class: "+r.getYPredicted());
/*************************************************************************/
        sentence = "Awesome Home Recipes";
        r = classifier.predict(sentence);
        
        System.out.println("Classifing sentence: \""+sentence+"\"");
        System.out.println("Predicted class: "+r.getYPredicted());
/*************************************************************************/
        sentence = " 9GAG has the best funny pics, GIFs, videos, memes, cute, wtf, geeky, cosplay photos on the web. 9GAG is your best source of happiness. Check out 9GAG now!";
        r = classifier.predict(sentence);
        
        System.out.println("Classifing sentence: \""+sentence+"\"");
        System.out.println("Predicted class: "+r.getYPredicted());
/*************************************************************************/
        sentence = "Engadget | Technology News, Advice and Features";
        r = classifier.predict(sentence);
        
        System.out.println("Classifing sentence: \""+sentence+"\"");
        System.out.println("Predicted class: "+r.getYPredicted());

/*************************************************************************/
        sentence = "Games at Miniclip.com - Play Free Online Games";
        r = classifier.predict(sentence);
        
        System.out.println("Classifing sentence: \""+sentence+"\"");
        System.out.println("Predicted class: "+r.getYPredicted());
/*************************************************************************/
        sentence = "TechCrunch - The latest technology news and information on startups";
        r = classifier.predict(sentence);
        
        System.out.println("Classifing sentence: \""+sentence+"\"");
        System.out.println("Predicted class: "+r.getYPredicted());
        //Clean up
        //--------
        
        //Erase the classifier. This removes all files.
        classifier.erase();
    }
    
}
